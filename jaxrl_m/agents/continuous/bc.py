import copy
from functools import partial
from typing import Any
import jax
import jax.numpy as jnp
from jaxrl_m.common.encoding import EncodingWrapper, GCEncodingWrapper
import flax
import flax.linen as nn
import optax

from jaxrl_m.common.typing import Batch
from jaxrl_m.common.typing import PRNGKey
from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.networks.actor_critic_nets import Policy
from jaxrl_m.networks.mlp import MLP, FilmFullMLP, ConcatFirstMLP


@partial(jax.jit, static_argnums=(1, 2))
def get_inputs(batch: Batch, goal_conditioned: bool, language_conditioned: bool):
    if goal_conditioned:
        observations = (batch["observations"], batch["goals"])
    else:
        observations = batch["observations"]
    if language_conditioned:
        return (observations, batch["prompts"])
    else:
        return observations


class BCAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    goal_conditioned: bool = nonpytree_field()
    language_conditioned: bool = nonpytree_field()
    lr_schedule: Any = nonpytree_field()

    @jax.jit
    def _get_inputs(self, batch: Batch):
        return get_inputs(batch, self.goal_conditioned, self.language_conditioned)

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        def loss_fn(params, rng):
            rng, key = jax.random.split(rng)
            dist = self.state.apply_fn(
                {"params": params},
                self._get_inputs(batch),
                temperature=1.0,
                train=True,
                rngs={"dropout": key},
                name="actor",
            )
            pi_actions = dist.mode()
            log_probs = dist.log_prob(batch["actions"])
            mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)
            actor_loss = -(log_probs).mean()
            actor_std = dist.stddev().mean(axis=1)

            return actor_loss, {
                "actor_loss": actor_loss,
                "mse": mse.mean(),
                "log_probs": log_probs.mean(),
                "pi_actions": pi_actions.mean(),
                "mean_std": actor_std.mean(),
                "max_std": actor_std.max(),
            }

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fn, pmap_axis=pmap_axis, has_aux=True
        )

        # log learning rates
        info["lr"] = self.lr_schedule(self.state.step)

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames="argmax")
    def sample_actions(
        self, batch: Batch, *, seed: PRNGKey, temperature: float = 1.0, argmax=False
    ) -> jnp.ndarray:
        dist = self.state.apply_fn(
            {"params": self.state.params},
            self._get_inputs(batch),
            temperature=temperature,
            name="actor",
        )
        if argmax:
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        return actions

    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        dist = self.state.apply_fn(
            {"params": self.state.params},
            self._get_inputs(batch),
            temperature=1.0,
            name="actor",
        )
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"])
        mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)

        return {
            "mse": mse,
            "log_probs": log_probs,
            "pi_actions": pi_actions,
        }

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        example_batch: Batch,
        # Model architecture
        encoder_def: nn.Module,
        goal_conditioned: bool = False,
        language_conditioned: bool = False,
        shared_goal_encoder: bool = True,
        early_goal_concat: bool = False,
        use_proprio: bool = False,
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": False,
            "state_dependent_std": False,
            "dropout": 0.0,
        },
        # Optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        decay_steps: int = 1000000,
    ):
        if goal_conditioned:
            if early_goal_concat:
                # passing None as the goal encoder causes early goal concat
                goal_encoder_def = None
            else:
                if shared_goal_encoder:
                    goal_encoder_def = encoder_def
                else:
                    goal_encoder_def = copy.deepcopy(encoder_def)

            encoder_def = GCEncodingWrapper(
                encoder=encoder_def,
                goal_encoder=goal_encoder_def,
                use_proprio=use_proprio,
                stop_gradient=False,
            )
        else:
            encoder_def = EncodingWrapper(
                encoder=encoder_def,
                use_proprio=use_proprio,
                stop_gradient=False,
            )

        if language_conditioned:
            # network = FilmFullMLP(**network_kwargs)
            network = ConcatFirstMLP(**network_kwargs)
        else:
            network_kwargs["activate_final"] = True
            network = MLP(**network_kwargs)

        networks = {
            "actor": Policy(
                encoder_def,
                network,
                action_dim=example_batch["actions"].shape[-1],
                language_conditioned=language_conditioned,
                **policy_kwargs
            )
        }

        model_def = ModuleDict(networks)

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=0.0,
        )
        tx = optax.adam(lr_schedule)

        rng, init_rng = jax.random.split(rng)
        inputs = get_inputs(example_batch, goal_conditioned, language_conditioned)
        params = model_def.init(init_rng, actor=[inputs])["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            rng=create_rng,
        )

        return cls(state, goal_conditioned, language_conditioned, lr_schedule)
