import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from utils.vorticity import velocity_to_vorticity_fwd
from utils.data_utils import helmholtz3d_exact_u, klein_gordon3d_exact_u
from utils.vorticity import vorx, vory, vorz
import seaborn as sns

import pdb


def _diffusion3d(args, apply_fn, params, test_data, result_dir, e, resol):
    print("visualizing solution...")

    nt = 11 # number of time steps to visualize
    t = jnp.linspace(0., 1., nt)
    x = jnp.linspace(-1., 1., resol)
    y = jnp.linspace(-1., 1., resol)
    xd, yd = jnp.meshgrid(x, y, indexing='ij')  # for 3-d surface plot
    tm, xm, ym = jnp.meshgrid(t, x, y, indexing='ij')
    if args.model == 'pinn':
        t = tm.reshape(-1, 1)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
    else:
        t = t.reshape(-1, 1)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

    u_ref = test_data[-1]
    ref_idx = 0

    os.makedirs(os.path.join(result_dir, f'vis/{e:05d}'), exist_ok=True)
    u = apply_fn(params, t, x, y)
    if args.model == 'pinn':
        u = u.reshape(nt, resol, resol)
        u_ref = u_ref.reshape(-1, resol, resol)

    for tt in range(nt):
        fig = plt.figure(figsize=(12, 6))

        # reference solution (hard-coded; must be modified if nt changes)
        ax1 = fig.add_subplot(121, projection='3d')
        im = ax1.plot_surface(xd, yd, u_ref[ref_idx], cmap='jet', linewidth=0, antialiased=False)
        ref_idx += 10
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u')
        ax1.set_title(f'Reference $u(x, y)$ at $t={tt*(1/(nt-1)):.1f}$', fontsize=6)
        ax1.set_zlim(jnp.min(u_ref), jnp.max(u_ref))

        # predicted solution
        ax2 = fig.add_subplot(122, projection='3d')
        im = ax2.plot_surface(xd, yd, u[tt], cmap='jet', linewidth=0, antialiased=False)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('u')
        ax2.set_title(f'Predicted $u(x, y)$ at $t={tt*(1/(nt-1)):.1f}$', fontsize=6)
        ax2.set_zlim(jnp.min(u_ref), jnp.max(u_ref))

        plt.savefig(os.path.join(result_dir, f'vis/{e:05d}/pred_{tt*(1/(nt-1)):.1f}.png'))
        plt.close()


def _helmholtz3d(args, apply_fn, params, result_dir, e, resol):
    print("visualizing solution...")

    x = jnp.linspace(-1., 1., resol)
    y = jnp.linspace(-1., 1., resol)
    z = jnp.linspace(-1., 1., resol)
    xm, ym, zm = jnp.meshgrid(x, y, z, indexing='ij')
    if args.model == 'pinn':
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        z = zm.reshape(-1, 1)
    else:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = z.reshape(-1, 1)

    u_ref = helmholtz3d_exact_u(args.a1, args.a2, args.a3, xm, ym, zm)

    os.makedirs(os.path.join(result_dir, f'vis/{e:05d}'), exist_ok=True)
    u_pred = apply_fn(params, x, y, z)
    if args.model == 'pinn':
        u_pred = u_pred.reshape(resol, resol, resol)
        u_ref = u_ref.reshape(resol, resol, resol)

    fig = plt.figure(figsize=(14, 5))

    # reference solution
    ax1 = fig.add_subplot(131, projection='3d')
    im = ax1.scatter(xm, ym, zm, c=u_ref, cmap = 'seismic', s=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title(f'Reference $u(x, y, z)$', fontsize=6)

    # predicted solution
    ax2 = fig.add_subplot(132, projection='3d')
    im = ax2.scatter(xm, ym, zm, c=u_pred, cmap = 'seismic', s=0.5, vmin=jnp.min(u_ref), vmax=jnp.max(u_ref))
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title(f'Predicted $u(x, y, z)$', fontsize=6)

    # absolute error
    ax3 = fig.add_subplot(133, projection='3d')
    im = ax3.scatter(xm, ym, zm, c=jnp.abs(u_ref-u_pred), cmap = 'seismic', s=0.5, vmin=jnp.min(u_ref), vmax=jnp.max(u_ref))
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.set_title(f'Absolute error', fontsize=6)

    cbar_ax = fig.add_axes([0.95, 0.3, 0.01, 0.4])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(os.path.join(result_dir, f'vis/{e:05d}/pred.png'))
    plt.close()


def _klein_gordon3d(args, apply_fn, params, result_dir, e, resol):
    print("visualizing solution...")

    t = jnp.linspace(0., 10., resol)
    x = jnp.linspace(-1., 1., resol)
    y = jnp.linspace(-1., 1., resol)
    tm, xm, ym = jnp.meshgrid(t, x, y, indexing='ij')
    if args.model == 'pinn':
        t = tm.reshape(-1, 1)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
    else:
        t = t.reshape(-1, 1)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

    u_ref = klein_gordon3d_exact_u(tm, xm, ym, args.k)

    os.makedirs(os.path.join(result_dir, f'vis/{e:05d}'), exist_ok=True)
    u_pred = apply_fn(params, t, x, y)
    if args.model == 'pinn':
        u_pred = u_pred.reshape(resol, resol, resol)
        u_ref = u_ref.reshape(resol, resol, resol)

    fig = plt.figure(figsize=(14, 5))

    # reference solution
    ax1 = fig.add_subplot(131, projection='3d')
    im = ax1.scatter(tm, xm, ym, c=u_ref, cmap = 'seismic', s=0.5)
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.set_zlabel('y')
    ax1.set_title(f'Reference $u(t, x, y)$', fontsize=6)

    # predicted solution
    ax2 = fig.add_subplot(132, projection='3d')
    im = ax2.scatter(tm, xm, ym, c=u_pred, cmap = 'seismic', s=0.5, vmin=jnp.min(u_ref), vmax=jnp.max(u_ref))
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_zlabel('y')
    ax2.set_title(f'Predicted $u(t, x, y)$', fontsize=6)

    # absolute error
    ax3 = fig.add_subplot(133, projection='3d')
    im = ax3.scatter(tm, xm, ym, c=jnp.abs(u_ref-u_pred), cmap = 'seismic', s=0.5, vmin=jnp.min(u_ref), vmax=jnp.max(u_ref))
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')
    ax3.set_zlabel('y')
    ax3.set_title(f'Absolute error', fontsize=6)

    cbar_ax = fig.add_axes([0.95, 0.3, 0.01, 0.4])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(os.path.join(result_dir, f'vis/{e:05d}/pred.png'))
    plt.close()


def _navier_stokes3d(apply_fn, params, test_data, result_dir, e):
    print("visualizing solution...")

    nt, nx, ny = test_data[0].shape[0], test_data[1].shape[0], test_data[2].shape[0]

    t = test_data[0][-1]
    t = jnp.expand_dims(t, axis=1)

    w_pred = velocity_to_vorticity_fwd(apply_fn, params, t, test_data[1], test_data[2])
    w_pred = w_pred.reshape(-1, nx, ny)
    w_ref = test_data[-1][-1]

    os.makedirs(os.path.join(result_dir, f'vis/{e:05d}'), exist_ok=True)

    fig = plt.figure(figsize=(14, 5))

    # reference solution
    ax1 = fig.add_subplot(131)
    im = ax1.imshow(w_ref, cmap='jet', extent=[0, 2*jnp.pi, 0, 2*jnp.pi], vmin=jnp.min(w_ref), vmax=jnp.max(w_ref))
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_title(f'Reference $\omega(t={jnp.round(t[0][0], 1):.2f}, x, y)$', fontsize=6)

    # predicted solution
    ax1 = fig.add_subplot(132)
    im = ax1.imshow(w_pred[0], cmap='jet', extent=[0, 2*jnp.pi, 0, 2*jnp.pi], vmin=jnp.min(w_ref), vmax=jnp.max(w_ref))
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_title(f'Predicted $\omega(t={jnp.round(t[0][0], 1):.2f}, x, y)$', fontsize=6)

    # absolute error
    ax1 = fig.add_subplot(133)
    im = ax1.imshow(jnp.abs(w_ref - w_pred[0]), cmap='jet', extent=[0, 2*jnp.pi, 0, 2*jnp.pi], vmin=jnp.min(w_ref), vmax=jnp.max(w_ref))
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_title(f'Asolute error', fontsize=6)

    cbar_ax = fig.add_axes([0.95, 0.3, 0.01, 0.4])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(os.path.join(result_dir, f'vis/{e:05d}/pred.png'))
    plt.close()



def _boussinesq_convection_flow_3d(apply_fn, params, test_data, result_dir, e):
    print("visualizing solution...")

    nt, nx, ny = test_data[0].shape[0], test_data[1].shape[0], test_data[2].shape[0]

    # n_lices = 10
    # dt = nt // n_lices
    # time_slices = [test_data[0][i*dt] for i in range(n_lices-1)]
    # # print(time_slices)
    #
    # for t_i in time_slices:

    # for t_i in [test_data[0][0], test_data[0][nt//2],test_data[0][-1]]:
    i=0
    t = []
    rho_pred, rho0_ref = [], []
    for t_i in [test_data[0][0], test_data[0][nt//3], test_data[0][2 * nt//3],test_data[0][-1]]:
        t += [jnp.expand_dims(t_i, axis=1)]

        # w_pred = velocity_to_vorticity_fwd(apply_fn, params, t, test_data[1], test_data[2])
        # w_pred = w_pred.reshape(-1, nx, ny)
        # w_ref = test_data[-1][-1]

        x = test_data[1]
        y = test_data[2]
        u0, v0, rho0_pred = apply_fn(params, t[i], x, y)
        rho_pred += [rho0_pred.reshape(-1, nx, ny)]
        rho0_ref += [test_data[-1][-1]]

        i += 1



    os.makedirs(os.path.join(result_dir, f'vis/{e:05d}'), exist_ok=True)

    # fig = plt.figure(figsize=(18, 9))
    fig, axs = plt.subplots(2, 4)

    x, y = jnp.meshgrid(x.ravel(), y.ravel(), indexing='ij')

    # reference solution
    # ax1 = fig.add_subplot(141, aspect='equal')
    # im = ax1.pcolor(x, y, rho0_ref, cmap='RdBu', vmin=jnp.min(rho0_ref), vmax=jnp.max(rho0_ref))
    levels = jnp.linspace(0.1, 5, 10)
    origin = 'lower'
    CS = axs[0, 0].contourf(x, y, rho_pred[0][0], levels,
                       origin=origin,
                       extend='both')
    CS2 = axs[0, 0].contour(CS, levels=CS.levels[::2], colors='r', origin=origin)
    axs[0, 0].clabel(CS2, fmt='%2.1f', colors='w', fontsize=11)
    # fig.colorbar(im)
    axs[0, 0].set_xlabel('$x$')
    axs[0, 0].set_ylabel('$y$')
    axs[0, 0].set_box_aspect(1)
    axs[0, 0].set_title(f'Cont.s $\\rho(t={jnp.round(t[0][0][0], 1):.2f}, x, y); ep. {e:.2f}$', fontsize=6)

    cmap = sns.color_palette('icefire', as_cmap=True)
    # predicted solution
    # ax1 = fig.add_subplot(241, aspect='equal')
    im = axs[1, 0].pcolor(x, y, rho_pred[0][0], cmap='rainbow', vmin=jnp.min(rho0_pred[0]), vmax=jnp.max(rho0_pred[0]))
    fig.colorbar(im,fraction=0.046, pad=0.04)
    axs[1, 0].set_xlabel('$x$')
    axs[1, 0].set_ylabel('$y$')
    axs[1, 0].set_box_aspect(1)
    axs[1, 0].set_title(f'Predicted $\\rho(t={jnp.round(t[0][0][0], 1):.2f}, x, y)$', fontsize=6)

    # reference solution
    # ax1 = fig.add_subplot(142, aspect='equal')
    # im = ax1.pcolor(x, y, rho0_ref, cmap='RdBu', vmin=jnp.min(rho0_ref), vmax=jnp.max(rho0_ref))
    levels = jnp.linspace(0.1, 5, 10)
    origin = 'lower'
    CS = axs[0, 1].contourf(x, y, rho_pred[1][0], levels,
                       origin=origin,
                       extend='both')
    CS2 = axs[0, 1].contour(CS, levels=CS.levels[::2], colors='r', origin=origin)
    axs[0, 1].clabel(CS2, fmt='%2.1f', colors='w', fontsize=11)
    # fig.colorbar(im)
    axs[0, 1].set_xlabel('$x$')
    axs[0, 1].set_ylabel('$y$')
    axs[0, 1].set_box_aspect(1)
    axs[0, 1].set_title(f'Cont.s $\\rho(t={jnp.round(t[1][0][0], 1):.2f}, x, y); ep. {e:.2f}$', fontsize=6)

    cmap = sns.color_palette('icefire', as_cmap=True)
    # predicted solution
    # ax1 = fig.add_subplot(242, aspect='equal')
    im = axs[1, 1].pcolor(x, y, rho_pred[1][0], cmap='rainbow', vmin=jnp.min(rho0_pred[0]), vmax=jnp.max(rho0_pred[0]))
    fig.colorbar(im,fraction=0.046, pad=0.04)
    axs[1, 1].set_xlabel('$x$')
    axs[1, 1].set_ylabel('$y$')
    axs[1, 1].set_box_aspect(1)
    axs[1, 1].set_title(f'Predicted $\\rho(t={jnp.round(t[1][0][0], 1):.2f}, x, y)$', fontsize=6)

    # reference solution
    # ax1 = fig.add_subplot(143, aspect='equal')
    # im = ax1.pcolor(x, y, rho0_ref, cmap='RdBu', vmin=jnp.min(rho0_ref), vmax=jnp.max(rho0_ref))
    levels = jnp.linspace(0.1, 5, 10)
    origin = 'lower'
    CS = axs[0, 2].contourf(x, y, rho_pred[2][0], levels,
                       origin=origin,
                       extend='both')
    CS2 = axs[0, 2].contour(CS, levels=CS.levels[::2], colors='r', origin=origin)
    axs[0, 2].clabel(CS2, fmt='%2.1f', colors='w', fontsize=11)
    # fig.colorbar(im)
    axs[0, 2].set_xlabel('$x$')
    axs[0, 2].set_ylabel('$y$')
    axs[0, 2].set_box_aspect(1)
    axs[0, 2].set_title(f'Cont.s $\\rho(t={jnp.round(t[2][0][0], 1):.2f}, x, y); ep. {e:.2f}$', fontsize=6)

    cmap = sns.color_palette('icefire', as_cmap=True)
    # predicted solution
    # ax1 = fig.add_subplot(243, aspect='equal')
    im = axs[1, 2].pcolor(x, y, rho_pred[2][0], cmap='rainbow', vmin=jnp.min(rho0_pred[0]), vmax=jnp.max(rho0_pred[0]))
    fig.colorbar(im,fraction=0.046, pad=0.04)
    axs[1, 2].set_xlabel('$x$')
    axs[1, 2].set_ylabel('$y$')
    axs[1, 2].set_box_aspect(1)
    axs[1, 2].set_title(f'Predicted $\\rho(t={jnp.round(t[2][0][0], 1):.2f}, x, y)$', fontsize=6)

    # reference solution
    # ax1 = fig.add_subplot(144, aspect='equal')
    # im = ax1.pcolor(x, y, rho0_ref, cmap='RdBu', vmin=jnp.min(rho0_ref), vmax=jnp.max(rho0_ref))
    levels = jnp.linspace(0.01, 5, 10)
    origin = 'lower'
    CS = axs[0, 3].contourf(x, y, rho_pred[3][0], levels,
                       origin=origin,
                       extend='both')
    CS2 = axs[0, 3].contour(CS, levels=CS.levels[::2], colors='r', origin=origin)
    axs[0, 3].clabel(CS2, fmt='%2.1f', colors='w', fontsize=11)
    # fig.colorbar(im)
    axs[0, 3].set_xlabel('$x$')
    axs[0, 3].set_ylabel('$y$')
    axs[0, 3].set_box_aspect(1)
    axs[0, 3].set_title(f'Cont.s $\\rho(t={jnp.round(t[3][0][0], 1):.2f}, x, y); ep. {e:.2f}$', fontsize=6)

    cmap = sns.color_palette('icefire', as_cmap=True)
    # predicted solution
    # ax1 = fig.add_subplot(244, aspect='equal')
    im = axs[1, 3].pcolor(x, y, rho_pred[3][0], cmap='rainbow', vmin=jnp.min(rho0_pred[0]), vmax=jnp.max(rho0_pred[0]))
    fig.colorbar(im,fraction=0.046, pad=0.04)
    axs[1, 3].set_xlabel('$x$')
    axs[1, 3].set_ylabel('$y$')
    axs[1, 3].set_box_aspect(1)
    axs[1, 3].set_title(f'Predicted $\\rho(t={jnp.round(t[3][0][0], 1):.2f}, x, y)$', fontsize=6)

    # # absolute error
    # error = jnp.abs(rho0_ref - rho0_pred[0])
    # ax1 = fig.add_subplot(331, aspect='equal')
    # im = ax1.pcolor(x, y, error, cmap='rainbow', vmin=jnp.min(error), vmax=jnp.max(error))
    # ax1.set_xlabel('$x$')
    # ax1.set_ylabel('$y$')
    # ax1.set_title(f'Asolute error', fontsize=15)
    #
    # # cbar_ax = fig.add_axes([0.95, 0.3, 0.01, 0.4])
    # fig.colorbar(im)

    plt.savefig(os.path.join(result_dir, f'vis/{e:05d}/pred_t{t[0][0]}.png'))
    plt.show()
    plt.close()


def _navier_stokes4d(apply_fn, params, test_data, result_dir, e):
    print("visualizing solution...")

    os.makedirs(os.path.join(result_dir, f'vis/{e:05d}'), exist_ok=True)

    fig = plt.figure(figsize=(30, 5))
    for t, sub in zip([0, 1, 2, 3, 4, 5], [161, 162, 163, 164, 165, 166]):
        t = jnp.array([[t]])
        x = jnp.linspace(0, 2*jnp.pi, 4).reshape(-1, 1)
        y = jnp.linspace(0, 2*jnp.pi, 30).reshape(-1, 1)
        z = jnp.linspace(0, 2*jnp.pi, 30).reshape(-1, 1)
        wx = vorx(apply_fn, params, t, x, y, z)
        wy = vory(apply_fn, params, t, x, y, z)
        wz = vorz(apply_fn, params, t, x, y, z)

        # c = jnp.sqrt(u_x**2 + u_y**2 + u_z**2)   # magnitude
        c = jnp.arctan2(wy, wz)    # zenith angle
        c = (c.ravel() - c.min()) / c.ptp()
        c = jnp.concatenate((c, jnp.repeat(c, 2)))
        c = plt.cm.plasma(c)

        x, y, z = jnp.meshgrid(jnp.squeeze(x), jnp.squeeze(y), jnp.squeeze(z), indexing='ij')

        ax = fig.add_subplot(sub, projection='3d')
        ax.quiver(x, y, z, jnp.squeeze(wx), jnp.squeeze(wy), jnp.squeeze(wz), length=0.1, colors=c, alpha=1, linewidth=0.7)
        plt.title(f't={jnp.squeeze(t)}')
    
    plt.savefig(os.path.join(result_dir, f'vis/{e:05d}/pred.png'))
    plt.close()


def show_solution(args, apply_fn, params, test_data, result_dir, e, resol=50):
    if args.equation == 'diffusion3d':
        _diffusion3d(args, apply_fn, params, test_data, result_dir, e, resol)
    elif args.equation == 'helmholtz3d':
        _helmholtz3d(args, apply_fn, params, result_dir, e, resol)
    elif args.equation == 'klein_gordon3d':
        _klein_gordon3d(args, apply_fn, params, result_dir, e, resol)
    elif args.equation == 'navier_stokes3d':
        _navier_stokes3d(apply_fn, params, test_data, result_dir, e)
    elif args.equation == 'Boussinesq_convection_flow_3d':
        _boussinesq_convection_flow_3d(apply_fn, params, test_data, result_dir, e)
    elif args.equation == 'navier_stokes4d':
        _navier_stokes4d(apply_fn, params, test_data, result_dir, e)
    else:
        raise NotImplementedError