import numpy as np
import matplotlib.pyplot as plt
import definitions as defs
import hm_dfield as hm

L = 2048
sigma = 6.0
X, Y = hm.real_grid(L)
KX, KY = hm.freq_grid(L)
G_hat = hm.fft_gaussian_kernel(L, sigma)
kmag = hm.vector_magnitude(L, alpha=1000.0, H=0.0, K_ratio=0.8)
k1, k2 = hm.perpendicular_basis(kmag, angle=0.0, L=L)

widths = [("pi", np.pi),
          ("pi/2", np.pi/2),
          ("pi/4", np.pi/4),
          ("pi/8", np.pi/8),
          ("pi/16", np.pi/16),
          ("pi/32", np.pi/32),
          ("pi/64", np.pi/64),
          ("pi/128", np.pi/128),
          ("pi/256", np.pi/256)]


def point_cloud_plot():
    fig, axs = plt.subplots(len(widths), 3, figsize=(16, 4 * len(widths)))

    for i, (label, w) in enumerate(widths):
        np.random.seed(0)
        _, phi = defs.generate_grf(alpha=1000.0, H=0.0, K_ratio=0.8, L=L, angle=0.0, angle_width=w/2)
        eta1 = hm.demodulate(phi, k1, G_hat, X, Y)
        D1 = hm.d_field(eta1, KX, KY)
        q1 = hm.winding_number(eta1)

        ys, xs = np.nonzero(q1)
        charges = q1[ys, xs]

        # rho1 full 2048*2048 image
        psi0 = np.sqrt(np.mean(np.abs(eta1)**2))
        rho_raw = D1 / (np.pi * psi0**2)
        vmax = np.percentile(np.abs(rho_raw), 99.5)
        axs[i, 0].imshow(rho_raw, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        axs[i, 0].set_title(f"{label}: $D_1/\\pi\\Psi_0^2$")
        axs[i, 0].axis("off")

        # rho_top: D1 * Gaussian approximation of 2D dirac delta function
        eps = 0.2 * psi0
        rho_top = D1 * np.exp(-np.abs(eta1)**2 / (2 * eps**2)) / (2 * np.pi * eps**2)
        vloc = np.percentile(np.abs(rho_top), 99.9)
        axs[i, 1].imshow(rho_top, cmap="RdBu_r", vmin=-vloc, vmax=vloc, origin="lower")
        axs[i, 1].set_title(f"{label}: $D_1 \\cdot \\delta_\\epsilon(\\eta_1)$")
        axs[i, 1].axis("off")

        # Defect Point Cloud
        axs[i, 2].scatter(xs[charges > 0], ys[charges > 0], c="red", s=12, label=f"+1 ({(charges>0).sum()})")
        axs[i, 2].scatter(xs[charges < 0], ys[charges < 0], c="blue", s=12, label=f"-1 ({(charges<0).sum()})")
        axs[i, 2].set_xlim(0, L); axs[i, 2].set_ylim(0, L)
        axs[i, 2].set_aspect("equal")
        axs[i, 2].set_title(f"{label}: Defect Point Cloud ($2048^2$ full image)\nTotal = {len(charges)} defects")
        axs[i, 2].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig("pointcloud.png", dpi=300)
    print("Saved pointcloud.png")


def structure_factor_space(mask):
    """Compute the structure factor of a binary mask"""
    pts = mask.astype(float)
    N = pts.sum()
    if N == 0:
        return None

    # fft of the binary mask = sum of complex exponentials at the defect locations
    rho_hat = np.fft.fft2(pts)

    # structure factor = |rho_hat|^2 / N
    S = np.abs(rho_hat)**2 / N
    S[0, 0] = 0  # set the DC component to zero to avoid the peak at k=0

    # shifting origin to the center
    return np.fft.fftshift(S)

def structure_factor_value(eta):
    """Computes S(k) for negative, positive, and total defects separately"""
    q = hm.winding_number(eta)
    pos_mask = (q > 0).astype(float)
    neg_mask = (q < 0).astype(float)

    S_pos = structure_factor_space(pos_mask)
    S_neg = structure_factor_space(neg_mask)
    S_total = structure_factor_space(pos_mask + neg_mask)

    return S_pos, S_neg, S_total

def structure_factor_plot():
    fig, axs = plt.subplots(len(widths), 3, figsize=(16, 4 * len(widths)))

    for i, (label, w) in enumerate(widths):
        np.random.seed(0)
        _, phi = defs.generate_grf(alpha=1000.0, H=0.0, K_ratio=0.8, L=L, angle=0.0, angle_width=w/2)
        eta1 = hm.demodulate(phi, k1, G_hat, X, Y)

        S_pos, S_neg, S_total = structure_factor_value(eta1)

        panels = [
            (S_pos, f"{label}: $S_+(\\mathbf{{k}})$"),
            (S_neg, f"{label}: $S_-(\\mathbf{{k}})$"),
            (S_total, f"{label}: $S_{{\\mathrm{{total}}}}(\\mathbf{{k}})$"),
        ]

        for j, (S, title) in enumerate(panels):
            ax = axs[i, j]
            vmax = np.percentile(S, 99)
            im = ax.imshow(S, cmap="hot", vmin=0, vmax=vmax, origin="lower")
            ax.set_title(title)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("structure_factor.png", dpi=150)
    print("Saved structure_factor.png")


if __name__ == "__main__":
    structure_factor_plot()
