import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
from scipy.spatial import ConvexHull

# ---------------------------------------------------------
# 1. Ultra-Compact Plot Styling
# ---------------------------------------------------------
plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 
    'figure.figsize': (8.5, 4.0), # Reduced height, widened for right-side legend
    'figure.dpi': 300, 'font.family': 'sans-serif'
})

np.random.seed(30)

# ---------------------------------------------------------
# 2. Refined Data Simulation
# ---------------------------------------------------------
mu_A, cov_A = [-2, 4], [[0.1, 0], [0, 0.1]] 
mu_B, cov_B = [2, 2], [[2, 1.8], [1.8, 2]] 

# Stream Snapshots
X_A1, X_B1 = np.random.multivariate_normal(mu_A, cov_A, 6), np.random.multivariate_normal(mu_B, cov_B, 6)
X_A2 = np.vstack([X_A1, np.random.multivariate_normal(mu_A, cov_A, 45)])
X_B2 = np.vstack([X_B1, np.random.multivariate_normal(mu_B, cov_B, 10)])
X_A3 = X_A2
X_B3 = np.vstack([X_B2, np.random.multivariate_normal(mu_B, cov_B, 150)])
datasets = [(X_A1, X_B1), (X_A2, X_B2), (X_A3, X_B3)]

alphas_A = [0.6, 0.9, 0.25]
alphas_B = [0.6, 0.25, 0.9]

# Predefined sizes
POINT_SIZES = {
    'small': 30,
    'medium': 60,
    'large': 120
}

coresets = {
    'Naive': [
        {'pts': [[-1.8, 3.8], [-2.1, 4.1], [1.8, 2.1], [2.1, 1.9]], 'w': [0.25]*4, 
         'sizes': ['small', 'small', 'small', 'small']},
        {'pts': [[-2.1, 3.7], [-2.2, 4.2], [-1.7, 4.0], [1.5, 1.5]], 'w': [0.25]*4, 
         'sizes': ['small', 'small', 'small', 'small']},
        {'pts': [[-2.1, 3.7], [-1.7, 4.0], [1.8, 1.8], [2.2, 2.2]], 'w': [0.25]*4, 
         'sizes': ['small', 'small', 'small', 'small']},
    ],
    'Proposed': [
        {'pts': [[-1.8, 3.8], [-2.1, 4.1], [1.8, 2.1], [2.1, 1.9]], 'w': [0.25]*4, 
         'sizes': ['small', 'small', 'small', 'small']},
        {'pts': [[-2.1, 3.9], [-1.8, 4.2], [0.5, 1.0], [3.5, 3.0]], 'w': [0.4, 0.4, 0.1, 0.1], 
         'sizes': ['medium', 'medium', 'small', 'small']},
        {'pts': [[-2.0, 4.0], [2.0, 2.0], [0.5, 0.5], [4.0, 3.5]], 'w': [0.1, 0.7, 0.1, 0.1], 
         'sizes': ['small', 'large', 'medium', 'medium']},
    ]
}

# ---------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
fig.subplots_adjust(left=0.08, right=0.82, bottom=0.18, top=0.95, wspace=0.08, hspace=0.1)

for r_idx, algo in enumerate(['Naive', 'Proposed']):
    for c_idx in range(3):
        ax = axes[r_idx, c_idx]
        xa, xb = datasets[c_idx]
        
        # Clean Dominance Indication: Subtle background colors instead of overlapping halos
        if c_idx == 1:
            ax.set_facecolor('#ffebe6') # Soft red background
        elif c_idx == 2:
            ax.set_facecolor('#e6f0ff') # Soft blue background

        # Background Data
        ax.scatter(xa[:,0], xa[:,1], c='tab:red', alpha=alphas_A[c_idx], s=15, edgecolors='none', zorder=2)
        ax.scatter(xb[:,0], xb[:,1], c='tab:blue', alpha=alphas_B[c_idx], s=15, edgecolors='none', zorder=2)
        
        # Coreset Visualization
        cs = coresets[algo][c_idx]
        pts = np.array(cs['pts'])
        weights = np.array(cs['w'])
        sizes_str = cs['sizes']
        
        # Calculate means for L2 Error
        mu_t = (np.mean(xa, axis=0)*alphas_A[c_idx] + np.mean(xb, axis=0)*alphas_B[c_idx]) / (alphas_A[c_idx]+alphas_B[c_idx])
        mu_hat = np.average(pts, axis=0, weights=weights)
        l2_err = np.linalg.norm(mu_t - mu_hat)
        
        # Convex Hull Span
        if len(pts) >= 3:
            hull = ConvexHull(pts)
            poly = plt.Polygon(pts[hull.vertices], facecolor='black', alpha=0.1, linestyle='--', linewidth=0.7, zorder=1)
            ax.add_patch(poly)
        
        # Error Vector & True Mean
        ax.plot([mu_t[0], mu_hat[0]], [mu_t[1], mu_hat[1]], color='#D62728', lw=1, zorder=11)
        ax.scatter(mu_t[0], mu_t[1], marker='*', color='gold', s=60, edgecolors='black', zorder=12)
        
        # Weighted Atoms mapping
        atom_sizes = [POINT_SIZES[s] for s in sizes_str]
        ax.scatter(pts[:,0], pts[:,1], c='black', s=atom_sizes, alpha=0.8, edgecolors='white', linewidths=0.5, zorder=15)

        # Quality Annotation
        ax.text(0.95, 0.05, f"$L_2$: {l2_err:.2f}", transform=ax.transAxes, fontsize=8, color='#D62728', ha='right', weight='bold')

        # Axis Formatting
        ax.set_xlim(-3.5, 5.5)
        ax.set_ylim(-1, 5.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linestyle=':', alpha=0.3)

# ---------------------------------------------------------
# 4. Descriptive Layout & Legend
# ---------------------------------------------------------
titles = [r"$t_1$: Stream Entry", r"$t_2$: A Dominant", r"$t_3$: B Dominates"]
for i in range(3):
    axes[1, i].set_xlabel(titles[i], fontweight='bold', labelpad=5)

axes[0,0].set_ylabel("Unweighted\n(Naive)", fontweight='bold', labelpad=8)
axes[1,0].set_ylabel("Proposed\n(Weighted)", fontweight='bold', labelpad=8)

# Add a time progression arrow across the bottom
plt.annotate('', xy=(0.80, 0.04), xytext=(0.10, 0.04),
             xycoords='figure fraction', textcoords='figure fraction',
             arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15))
fig.text(0.45, 0.05, "Time", ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

# Vertical Legend on the right (Geometric Span moved to the top)
legend_el = [
    patches.Patch(facecolor='black', alpha=0.1, label='Geometric Span'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red', markersize=8, label='Concept A'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=8, label='Concept B'),
    mlines.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markeredgecolor='black', markersize=9, label=r'True Mean $\mu_t$'),
    mlines.Line2D([0], [0], color='#D62728', lw=1.5, label=r'$L_2$ Error')
]
fig.legend(handles=legend_el, loc='center left', bbox_to_anchor=(0.83, 0.5), ncol=1, frameon=True, fontsize=9)

plt.savefig("refined_coreset_research.pdf", bbox_inches='tight')
plt.show()
