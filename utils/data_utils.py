from functools import partial

import jax
import jax.numpy as jnp


# 3d time-independent helmholtz exact u
@partial(jax.jit, static_argnums=(0, 1, 2,))
def helmholtz3d_exact_u(a1, a2, a3, x, y, z):
    return jnp.sin(a1*jnp.pi*x) * jnp.sin(a2*jnp.pi*y) * jnp.sin(a3*jnp.pi*z)


# 3d time-independent helmholtz source term
@partial(jax.jit, static_argnums=(0, 1, 2,))
def helmholtz3d_source_term(a1, a2, a3, x, y, z, lda=1.):
    u_gt = helmholtz3d_exact_u(a1, a2, a3, x, y, z)
    uxx = -(a1*jnp.pi)**2 * u_gt
    uyy = -(a2*jnp.pi)**2 * u_gt
    uzz = -(a3*jnp.pi)**2 * u_gt
    return uxx + uyy + uzz + lda*u_gt


# 2d time-dependent klein-gordon exact u
def klein_gordon3d_exact_u(t, x, y, k):
    return (x + y) * jnp.cos(k * t) + (x * y) * jnp.sin(k * t)


# 2d time-dependent klein-gordon source term
def klein_gordon3d_source_term(t, x, y, k):
    u = klein_gordon3d_exact_u(t, x, y, k)
    return u**2 - (k**2)*u


# 2d time-dependent Boussinesq_convection_flow_3d
def Boussinesq_convection_flow_3d__initialvalue(t, x, y):

    rho1 = jnp.zeros_like(y)
    rho2 = jnp.zeros_like(y)
    R1 = jnp.sqrt(x ** 2 + (y - jnp.pi) ** 2)
    R2 = jnp.abs(x - 2 * jnp.pi)
    R3 = 1.95 * jnp.pi
    rho1[jnp.abs(R1) < jnp.pi] = jnp.exp(1 - jnp.pi ** 2 / (jnp.pi ** 2 - R1[jnp.abs(R1) < jnp.pi] ** 2))
    rho2[R2 < R3] = jnp.exp(1 - R3 ** 2 / (R3 ** 2 - R2[R2 < R3] ** 2))
    R = 50 * rho1 * rho2 * (1 - rho1)

    return R*0, R*0, R*0, R


# 3d time-dependent klein-gordon exact u
def klein_gordon4d_exact_u(t, x, y, z, k):
    return (x + y + z) * jnp.cos(k*t) + (x * y * z) * jnp.sin(k*t)


# 3d time-dependent klein-gordon source term
def klein_gordon4d_source_term(t, x, y, z, k):
    u = klein_gordon4d_exact_u(t, x, y, z, k)
    return u**2 - (k**2)*u


# 3d time-dependent navier-stokes forcing term
def navier_stokes4d_forcing_term(t, x, y, z, nu):
    # forcing terms in the PDE
    # f_x = -24*jnp.exp(-18*nu*t)*jnp.sin(2*y)*jnp.cos(2*y)*jnp.sin(z)*jnp.cos(z)
    f_x = -6*jnp.exp(-18*nu*t)*jnp.sin(4*y)*jnp.sin(2*z)
    # f_y = -24*jnp.exp(-18*nu*t)*jnp.sin(2*x)*jnp.cos(2*x)*jnp.sin(z)*jnp.cos(z)
    f_y = -6*jnp.exp(-18*nu*t)*jnp.sin(4*x)*jnp.sin(2*z)
    # f_z = 24*jnp.exp(-18*nu*t)*jnp.sin(2*x)*jnp.cos(2*x)*jnp.sin(2*y)*jnp.cos(2*y)
    f_z = 6*jnp.exp(-18*nu*t)*jnp.sin(4*x)*jnp.sin(4*y)
    return f_x, f_y, f_z


# 3d time-dependent navier-stokes exact vorticity
def navier_stokes4d_exact_w(t, x, y, z, nu):
    # analytic form of vorticity
    w_x = -3*jnp.exp(-9*nu*t)*jnp.sin(2*x)*jnp.cos(2*y)*jnp.cos(z)
    w_y = 6*jnp.exp(-9*nu*t)*jnp.cos(2*x)*jnp.sin(2*y)*jnp.cos(z)
    w_z = -6*jnp.exp(-9*nu*t)*jnp.cos(2*x)*jnp.cos(2*y)*jnp.sin(z)
    return w_x, w_y, w_z


# 3d time-dependent navier-stokes exact velocity
def navier_stokes4d_exact_u(t, x, y, z, nu):
    # analytic form of velocity
    u_x = 2*jnp.exp(-9*nu*t)*jnp.cos(2*x)*jnp.sin(2*y)*jnp.sin(z)
    u_y = -1*jnp.exp(-9*nu*t)*jnp.sin(2*x)*jnp.cos(2*y)*jnp.sin(z)
    u_z = -2*jnp.exp(-9*nu*t)*jnp.sin(2*x)*jnp.sin(2*y)*jnp.cos(z)
    return u_x, u_y, u_z