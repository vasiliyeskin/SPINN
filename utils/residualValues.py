from utils.vorticity import velocity_to_vorticity_fwd
import jax.numpy as jnp
from jax import jvp
import jax


def get_residuals(apply_fn, params, t, x, y):
    # compute [u, v]
    uv = apply_fn(params, t, x, y)

    vec_t = jnp.ones(t.shape)
    vec_xy = jnp.ones(x.shape)
    w_t = jvp(
        lambda t: velocity_to_vorticity_fwd(apply_fn, params, t, x, y),
        (t,),
        (vec_t,)
    )[1]
    w_x = jvp(
        lambda x: velocity_to_vorticity_fwd(apply_fn, params, t, x, y),
        (x,),
        (vec_xy,)
    )[1]
    w_y = jvp(
        lambda y: velocity_to_vorticity_fwd(apply_fn, params, t, x, y),
        (y,),
        (vec_xy,)
    )[1]
    rho_t = jvp(lambda t: apply_fn(params, t, x, y)[2], (t,), (vec_t,))[1]
    rho_x = jvp(lambda x: apply_fn(params, t, x, y)[2], (x,), (vec_xy,))[1]
    rho_y = jvp(lambda y: apply_fn(params, t, x, y)[2], (y,), (vec_xy,))[1]


    # PDE constraint
    R_rho = rho_t + uv[0] * rho_x + uv[1] * rho_y
    abs_rho = jnp.abs(R_rho)

    R_w = w_t + uv[0] * w_x + uv[1] * w_y - rho_x
    abs_w = jnp.abs(R_w)

    # incompressible fluid constraint
    u_x = jvp(lambda x: apply_fn(params, t, x, y)[0], (x,), (vec_xy,))[1]
    v_y = jvp(lambda y: apply_fn(params, t, x, y)[1], (y,), (vec_xy,))[1]
    R_c = u_x + v_y
    abs_c = jnp.abs(R_c)


    return abs_rho, abs_w, abs_c