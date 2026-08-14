"""
Gera figuras de publicação para sbrt2026.tex.
Salva PDF + PNG em figures/.

Rodar:
  cd "/mnt/c/.../Redes Neurais/results/latex"
  python generate_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from pathlib import Path

# ── Estilo IEEE / SBRT ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.size':          9,
    'axes.titlesize':     9,
    'axes.labelsize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    7.0,
    'lines.linewidth':    1.8,
    'lines.markersize':   5,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
})

OUT_DIR = Path(__file__).parent / 'figures'
OUT_DIR.mkdir(exist_ok=True)

# ── Dados finais (todos os métodos, pós-ablação D3F) ─────────────────────────
SNR = [0, 5, 10, 15, 20, 25, 30]

classical    = [0.081, 0.587, 0.999, 1.000, 1.000, 1.000, 1.000]
dnn_sigmoid  = [0.842, 0.994, 1.000, 1.000, 1.000, 1.000, 1.000]
cnn4feat     = [0.871, 0.995, 1.000, 1.000, 1.000, 1.000, 1.000]
svm_linear   = [0.305, 0.821, 0.997, 1.000, 1.000, 1.000, 1.000]
gmm_lrt      = [0.340, 0.718, 0.934, 1.000, 1.000, 1.000, 1.000]
cnn_1d       = [0.000, 0.000, 0.000, 0.896, 1.000, 1.000, 1.000]  # retrained
cnn_svm      = [0.000, 0.000, 0.000, 0.079, 0.988, 1.000, 1.000]  # corrected
dnn_logit    = [0.297, 0.821, 1.000, 1.000, 1.000, 1.000, 1.000]
cnn4feat_svm = [0.256, 0.971, 1.000, 1.000, 1.000, 1.000, 1.000]
dnn_1024     = [0.000, 0.000, 0.000, 0.000, 0.000, 0.998, 1.000]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: block-diagram box + arrow (used in architecture figures)
# ─────────────────────────────────────────────────────────────────────────────
def add_box(ax, cx, cy, w, h, l1, l2='',
            fc='#DBEAFE', fs1=6.8, fs2=4.8):
    """Rounded box at center (cx, cy), main label l1, subtitle l2."""
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.04',
        facecolor=fc, edgecolor='black', linewidth=1.1, zorder=3,
    )
    ax.add_patch(rect)
    if l2:
        ax.text(cx, cy + h * 0.18, l1, ha='center', va='center',
                fontsize=fs1, fontweight='bold', color='black', zorder=4)
        ax.text(cx, cy - h * 0.22, l2, ha='center', va='center',
                fontsize=fs2, color='#222222', zorder=4)
    else:
        ax.text(cx, cy, l1, ha='center', va='center',
                fontsize=fs1, fontweight='bold', color='black', zorder=4)


def add_arrow(ax, x1, x2, cy, lw=1.0):
    if x2 > x1 + 0.01:
        ax.annotate(
            '', xy=(x2, cy), xytext=(x1, cy),
            arrowprops=dict(arrowstyle='->', color='black',
                            lw=lw, mutation_scale=9),
            zorder=5,
        )


def save_fig(fig, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(str(OUT_DIR / f'{stem}.{ext}'))
    plt.close(fig)
    print(f'Saved: {stem}.pdf / .png')


# =============================================================================
# FIGURA 1 — Arquitetura da 4-feat DNN  (fig_arch_dnn)
#
# Fluxo: [f ∈ R^4] → Dense(256) → Dense(128) → Dense(64) → σ
# Cada camada densa inclui Batch Normalization, ReLU e Dropout.
# =============================================================================
def make_dnn_arch():
    fig, ax = plt.subplots(figsize=(3.45, 1.25))

    CY = 0.76   # vertical center of all boxes
    BH = 0.70   # box height
    GAP = 0.28  # horizontal gap between boxes (for arrows)
    M   = 0.15  # left margin

    # Each entry: (main label, subtitle, box width, face color)
    BLOCKS = [
        (r'$\mathbf{f}\!\in\!\mathbb{R}^4$',
         r'$|\tau|,\hat{h},\widehat{\rm SNR},E$',
         1.35, '#FFFBE6'),
        ('Dense(256)', 'BN+ReLU+Drop', 1.25, '#DBEAFE'),
        ('Dense(128)', 'BN+ReLU+Drop', 1.25, '#DBEAFE'),
        ('Dense(64)',  'BN+ReLU+Drop', 1.25, '#DBEAFE'),
        (r'$\sigma(\cdot)$', 'escore D3F', 1.05, '#FEE2D5'),
    ]

    total_w = M + sum(b[2] for b in BLOCKS) + GAP * (len(BLOCKS) - 1) + M
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, 1.52)
    ax.axis('off')

    x = M
    prev_right = None
    for (l1, l2, w, fc) in BLOCKS:
        cx = x + w / 2
        add_box(ax, cx, CY, w, BH, l1, l2, fc=fc)
        if prev_right is not None:
            add_arrow(ax, prev_right + 0.02, cx - w / 2 - 0.02, CY)
        prev_right = cx + w / 2
        x += w + GAP

    fig.tight_layout(pad=0.05)
    save_fig(fig, 'fig_arch_dnn')


# =============================================================================
# FIGURA 2 — Arquitetura da 1D CNN  (fig_arch_cnn)
#
# Fluxo: z_eq (1024×1) → Conv32 → Conv64 → Conv128 → GAP → Dense(64) → σ
# Cada bloco convolucional é seguido de BN, ReLU e MaxPooling(4).
# GAP realiza média global ao longo do tempo → vetor de 128 dim.
# =============================================================================
def make_cnn_arch():
    fig, ax = plt.subplots(figsize=(3.45, 1.30))

    CY  = 0.78   # vertical center
    BH  = 0.70   # box height — same as DNN figure
    GAP = 0.26   # gap between box edges (arrow space)
    M   = 0.15   # left/right margin

    # (main_label, subtitle, width, facecolor)
    # Widths sized for text at fs1=4.8 / fs2=3.5
    # Colors: pale yellow (I/O), blue gradient (Conv/Dense), gray (GAP)
    BLOCKS = [
        (r'$\mathbf{z}_{eq}$', r'$1024{\times}1$', 0.88, '#FFFDE7'),
        ('Conv(32)',  r'$k\!=\!15$',                0.92, '#DBEAFE'),
        ('Conv(64)',  r'$k\!=\!9$',                 0.92, '#BFDBFE'),
        ('Conv(128)', r'$k\!=\!5$',                1.00, '#93C5FD'),
        ('GAP',       '128-d',                      0.72, '#E5E7EB'),
        ('Dense(64)', 'ReLU',                       0.92, '#DBEAFE'),
        (r'$\sigma$', 'score',                      0.62, '#FFFDE7'),
    ]

    total_w = M + sum(b[2] for b in BLOCKS) + GAP * (len(BLOCKS) - 1) + M
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, 1.56)
    ax.axis('off')

    x = M
    prev_right = None
    for (l1, l2, w, fc) in BLOCKS:
        cx = x + w / 2
        add_box(ax, cx, CY, w, BH, l1, l2, fc=fc, fs1=4.8, fs2=3.5)
        if prev_right is not None:
            add_arrow(ax, prev_right + 0.03, cx - w / 2 - 0.03, CY)
        prev_right = cx + w / 2
        x += w + GAP

    fig.tight_layout(pad=0.05)
    save_fig(fig, 'fig_arch_cnn')


# =============================================================================
# FIGURA 3 — PD vs SNR, 6 métodos-chave para comparação estruturada
#
# Grupo 4-feat (sólidas): DNN, CNN-4feat, SVM linear, correlador clássico.
#   → Responde: dentro de quem tem |τ|, precisamos de rede neural?
# Grupo z_eq (tracejada/pontilhada): 1D CNN, DNN-1024 (ablação de entrada).
#   → Mostra que sem acesso explícito à tag, mesmo DNNs falham abaixo 15 dB.
# =============================================================================
def make_pd_snr():
    fig, ax = plt.subplots(figsize=(3.45, 2.90))

    # ── Grupo 4-feat: têm acesso ao correlador |τ| (linhas sólidas) ──────────
    ax.plot(SNR, dnn_sigmoid, 'k-o',
            lw=2.2, ms=6, label='4-feat DNN [proposto]', zorder=6)
    ax.plot(SNR, cnn4feat, '-s',
            color='#333333', lw=1.8, ms=5, label='CNN-4feat', zorder=5)
    ax.plot(SNR, svm_linear, 'r-D',
            lw=1.6, ms=5, label='SVM linear (4-feat)')
    ax.plot(SNR, classical, 'b-^',
            lw=1.6, ms=5, label='Correlador clássico')

    # ── Grupo z_eq: sem acesso explícito à tag (tracejada/pontilhada) ────────
    ax.plot(SNR, cnn_1d, 'm--v',
            lw=1.6, ms=5, label=r'1D CNN ($\mathbf{z}_{eq}$)', alpha=0.9)
    ax.plot(SNR, dnn_1024, ':x',
            color='#555555', lw=1.4, ms=5,
            label=r'DNN-1024 ($\mathbf{z}_{eq}$, ablação)', alpha=0.85)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(r'$P_D$')
    ax.set_xlim(-1, 31)
    ax.set_ylim(-0.04, 1.10)
    ax.set_xticks(SNR)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.45, color='gray')
    ax.grid(True, which='minor', linestyle=':',  linewidth=0.4, alpha=0.25, color='gray')
    ax.legend(loc='lower right', framealpha=0.93, edgecolor='gray', fontsize=6.5)

    fig.tight_layout()
    save_fig(fig, 'fig_pd_snr')


# =============================================================================
# (Figura auxiliar — não publicada, para referência interna)
# fig_pd_snr_all: todos os métodos incluindo ablações
# =============================================================================
def make_pd_snr_all():
    fig, ax = plt.subplots(figsize=(3.45, 2.90))

    ax.plot(SNR, dnn_sigmoid,  'k-o',   lw=2.0, ms=5, label='4-feat DNN (sigmoid)', zorder=6)
    ax.plot(SNR, cnn4feat,     '-s',    lw=1.6, ms=5, color='#444444', label='CNN-4feat')
    ax.plot(SNR, classical,    'b-^',   lw=1.6, ms=5, label='Clássico')
    ax.plot(SNR, svm_linear,   'r-D',   lw=1.4, ms=5, label='SVM linear (4-feat)')
    ax.plot(SNR, gmm_lrt,      'g-P',   lw=1.4, ms=5, label='GMM-LRT')
    ax.plot(SNR, dnn_logit,    'k--o',  lw=1.2, ms=4, alpha=0.7, label='4-feat DNN (logit)')
    ax.plot(SNR, cnn_1d,       'm--v',  lw=1.4, ms=5, label=r'1D CNN ($z_{eq}$)')
    ax.plot(SNR, cnn_svm,      'c--*',  lw=1.2, ms=5, alpha=0.85, label=r'CNN-SVM ($z_{eq}$)')
    ax.plot(SNR, cnn4feat_svm, '--P',   lw=1.2, ms=5, color='#8B5CF6', label='CNN-4feat SVM')
    ax.plot(SNR, dnn_1024,     ':x',    lw=1.2, ms=5, color='#9CA3AF', label=r'DNN-1024 ($z_{eq}$)')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(r'$P_D$')
    ax.set_xlim(-1, 31)
    ax.set_ylim(-0.04, 1.10)
    ax.set_xticks(SNR)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.4, color='gray')
    ax.legend(loc='lower right', fontsize=5.8, framealpha=0.93, ncol=1)

    fig.tight_layout()
    save_fig(fig, 'fig_pd_snr_all')


# =============================================================================
# FIGURA 4 — Ablação de características: PD vs SNR  (fig_ablation_pd_snr)
#
# Mostra o papel de cada característica auxiliar.
# Contraste central: SNR nominal (V7) vs estimativas instantâneas (V6, V8).
# =============================================================================
def make_ablation_pd_snr():
    import json, pathlib

    json_path = pathlib.Path(__file__).parent.parent / 'data' / 'nn15_feature_ablation.json'
    with open(str(json_path)) as f:
        ab = json.load(f)

    def pd_list(key):
        return [ab[key]['pd_vs_snr'][str(s)] for s in SNR]

    fig, ax = plt.subplots(figsize=(3.45, 2.90))

    # Reference baseline
    ax.plot(SNR, classical, 'b-^', lw=1.4, ms=5, alpha=0.7,
            label='Correlador clássico')

    # V5: |τ| only — floor
    ax.plot(SNR, pd_list('V5_tau_only'), 'k:s', lw=1.3, ms=4, alpha=0.65,
            label=r'V5: $[|\tau|]$ (1 feat.)')

    # V7: nominal SNR — weakest 2-feat
    ax.plot(SNR, pd_list('V7_tau_snr'), 'r--D', lw=1.5, ms=5,
            label=r'V7: $[|\tau|,\,\overline{\mathrm{SNR}}]$ (nominal)')

    # V6: instantaneous ĥ
    ax.plot(SNR, pd_list('V6_tau_h'), 'g-o', lw=1.6, ms=5,
            label=r'V6: $[|\tau|,\,\hat{h}]$ (instantâneo)')

    # V8: instantaneous E — best 2-feat (bold)
    ax.plot(SNR, pd_list('V8_tau_energy'), 'm-v', lw=2.1, ms=6,
            label=r'V8: $[|\tau|,\,E]$ (instantâneo) [min]', zorder=6)

    # V1: 4-feat baseline for reference
    ax.plot(SNR, pd_list('V1_baseline'), 'k-o', lw=1.6, ms=5,
            alpha=0.55, label='V1: 4 feat. (linha de base)',
            dashes=(4, 2))

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(r'$P_D$')
    ax.set_xlim(-1, 31)
    ax.set_ylim(-0.04, 1.10)
    ax.set_xticks(SNR)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.45, color='gray')
    ax.grid(True, which='minor', linestyle=':',  linewidth=0.4, alpha=0.25, color='gray')
    ax.legend(loc='lower right', framealpha=0.93, edgecolor='gray', fontsize=6.0)

    # Annotation: highlight the 0 dB gap between V7 and V8
    ax.annotate('', xy=(0, pd_list('V8_tau_energy')[0]),
                xytext=(0, pd_list('V7_tau_snr')[0]),
                arrowprops=dict(arrowstyle='<->', color='#AA0000', lw=1.1))
    ax.text(0.45, (pd_list('V8_tau_energy')[0] + pd_list('V7_tau_snr')[0]) / 2,
            'inst. vs\nnominal', fontsize=5.5, color='#AA0000', va='center')

    fig.tight_layout()
    save_fig(fig, 'fig_ablation_pd_snr')


# =============================================================================
# FIGURA 5 — Diagrama de sistema PLA  (fig_system_diagram)
#
# Fluxo completo: Alice → Canal Rayleigh → Bob (extração → DNN/CNN → D3F).
# Eva aparece como transmissor alternativo (sem tag), seta vermelha diagonal.
# =============================================================================
def make_system_diagram():
    # 7" wide so 1 data-unit ≈ 1 inch — text fits comfortably at normal pt sizes.
    # Colors: only gray shades + pastel-blue gradient.
    fig, ax = plt.subplots(figsize=(7.0, 2.30))
    ax.set_xlim(0, 7.0)
    ax.set_ylim(0, 2.30)
    ax.axis('off')

    CG1 = '#F3F4F6'  # very light gray  – Alice
    CG2 = '#E5E7EB'  # light gray       – channel
    CG3 = '#D1D5DB'  # medium gray      – Eve
    CB1 = '#DBEAFE'  # pastel blue      – extraction
    CB2 = '#BFDBFE'  # pastel blue+     – NN
    CB3 = '#93C5FD'  # pastel blue++    – decision
    DARK = '#374151'

    Y_MAIN = 1.48
    Y_EVE  = 0.44
    BH     = 0.88    # main box height
    BHE    = 0.70    # Eve box height
    W      = 1.00    # uniform block width
    GAP    = 0.30    # gap between block edges
    M      = 0.35    # left/right margin
    FS1, FS2 = 7.5, 6.0

    def sbox(cx, cy, w, h, l1, l2='', fc='white', fs1=None, fs2=None):
        _fs1 = fs1 if fs1 is not None else FS1
        _fs2 = fs2 if fs2 is not None else FS2
        ax.add_patch(FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle='round,pad=0.04',
            facecolor=fc, edgecolor=DARK, linewidth=0.85, zorder=3))
        lines = l1.split('\n')
        if len(lines) == 2:
            # 2-line title + optional subtitle
            if l2:
                ax.text(cx, cy + h*0.28, lines[0], ha='center', va='center',
                        fontsize=_fs1, fontweight='bold', color=DARK, zorder=4)
                ax.text(cx, cy + h*0.07, lines[1], ha='center', va='center',
                        fontsize=_fs1, fontweight='bold', color=DARK, zorder=4)
                ax.text(cx, cy - h*0.28, l2, ha='center', va='center',
                        fontsize=_fs2, color='#4B5563', zorder=4)
            else:
                ax.text(cx, cy + h*0.12, lines[0], ha='center', va='center',
                        fontsize=_fs1, fontweight='bold', color=DARK, zorder=4)
                ax.text(cx, cy - h*0.12, lines[1], ha='center', va='center',
                        fontsize=_fs1, fontweight='bold', color=DARK, zorder=4)
        elif l2:
            ax.text(cx, cy + h*0.16, l1, ha='center', va='center',
                    fontsize=_fs1, fontweight='bold', color=DARK, zorder=4)
            ax.text(cx, cy - h*0.22, l2, ha='center', va='center',
                    fontsize=_fs2, color='#4B5563', zorder=4)
        else:
            ax.text(cx, cy, l1, ha='center', va='center',
                    fontsize=_fs1, fontweight='bold', color=DARK, zorder=4)

    def arr(x1, y1, x2, y2, lw=1.0, cs=''):
        kw = dict(arrowstyle='->', color=DARK, lw=lw, mutation_scale=9)
        if cs:
            kw['connectionstyle'] = cs
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=kw, zorder=5)

    # Block centres evenly spaced (M + W/2, then +W+GAP each step)
    cxs = [M + W/2 + i*(W + GAP) for i in range(5)]
    AX, CHX, EXX, NNX, DCX = cxs
    PAD = 0.06

    # ── blocks ───────────────────────────────────────────────────────────────
    sbox(AX,  Y_MAIN, W, BH,
         'Alice (Tx)', r'$x^i = \rho_s s^i + \rho_t t^i$', fc=CG1)
    sbox(CHX, Y_MAIN, W, BH,
         'Canal Rayleigh', r'$y^i = h^i x^i + w^i$', fc=CG2)
    sbox(EXX, Y_MAIN, W, BH,
         'Extração de\ncaracterísticas de canal',
         r'$|\tau|,\,\hat{h},\,E,\,\overline{\rm SNR},\,\ldots$',
         fc=CB1, fs1=5.8, fs2=5.2)
    sbox(NNX, Y_MAIN, W, BH,
         'DNN / CNN', r'$\hat{p} \in [0,1]$', fc=CB2)
    sbox(DCX, Y_MAIN, W, BH,
         'D3F', r'$\mathcal{H}_1 / \mathcal{H}_0$', fc=CB3)
    sbox(CHX, Y_EVE,  W, BHE,
         'Eva (Intrusa)', r'$\tilde{x}^i$ (sem tag)', fc=CG3)

    # ── main-flow arrows ─────────────────────────────────────────────────────
    for i in range(4):
        arr(cxs[i] + W/2 + PAD, Y_MAIN, cxs[i+1] - W/2 - PAD, Y_MAIN)

    # signal labels above first two arrows
    ax.text((AX  + W/2 + CHX - W/2) / 2, Y_MAIN + BH/2 + 0.06,
            r'$x^i$', ha='center', va='bottom', fontsize=6.5, color=DARK)
    ax.text((CHX + W/2 + EXX - W/2) / 2, Y_MAIN + BH/2 + 0.06,
            r'$y^i$', ha='center', va='bottom', fontsize=6.5, color=DARK)

    # Channel → Eve (down)
    arr(CHX, Y_MAIN - BH/2 - PAD, CHX, Y_EVE + BHE/2 + PAD)

    # Eve → Extraction (diagonal dark-gray — spoofing attempt)
    arr(CHX + W/2 + PAD, Y_EVE,
        EXX - W/2 - PAD, Y_MAIN - BH/2 + 0.08,
        lw=0.85, cs='arc3,rad=-0.12')
    ax.text((CHX + W/2 + EXX - W/2) / 2 + 0.05, (Y_EVE + Y_MAIN - BH/2) / 2,
            r'$\tilde{x}^i$', ha='left', va='center',
            fontsize=6.0, color='#6B7280', fontstyle='italic')

    # ── Bob's dashed group border ─────────────────────────────────────────────
    bx0 = EXX - W/2 - 0.14
    bx1 = DCX + W/2 + 0.14
    by0 = Y_MAIN - BH/2 - 0.16
    by1 = Y_MAIN + BH/2 + 0.26
    ax.add_patch(mpatches.Rectangle(
        (bx0, by0), bx1 - bx0, by1 - by0,
        fill=False, edgecolor='#2563EB',
        linewidth=0.70, linestyle='--', zorder=1))
    ax.text((bx0 + bx1) / 2, by1 + 0.05, 'Bob (Receptor)',
            ha='center', va='bottom', fontsize=6.5,
            color='#1D4ED8', fontstyle='italic')

    fig.tight_layout(pad=0.05)
    save_fig(fig, 'fig_system_diagram')


if __name__ == '__main__':
    make_dnn_arch()
    make_cnn_arch()
    make_pd_snr()
    make_pd_snr_all()
    make_ablation_pd_snr()
    make_system_diagram()
    print(f'\nTodas as figuras salvas em: {OUT_DIR}')
