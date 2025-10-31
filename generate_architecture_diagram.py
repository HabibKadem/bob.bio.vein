#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
Generate architecture diagram for CNN+ViT model for dorsal hand vein recognition
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.5, 'CNN+ViT Hybrid Model Architecture', 
        ha='center', va='center', fontsize=18, fontweight='bold')
ax.text(7, 9.1, 'for Dorsal Hand Vein Recognition (138 classes)', 
        ha='center', va='center', fontsize=12, style='italic')

# Define colors
color_input = '#E8F4F8'
color_cnn = '#B3D9FF'
color_vit = '#FFD9B3'
color_output = '#D4EDDA'
color_arrow = '#666666'

y_pos = 8.0

# Input Layer
input_box = FancyBboxPatch((0.5, y_pos), 2, 0.6, 
                           boxstyle="round,pad=0.05", 
                           facecolor=color_input, 
                           edgecolor='black', linewidth=2)
ax.add_patch(input_box)
ax.text(1.5, y_pos + 0.3, 'Input Image', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(1.5, y_pos + 0.05, '224×224×1', ha='center', va='center', fontsize=8)

# Arrow
arrow1 = FancyArrowPatch((2.5, y_pos + 0.3), (3.2, y_pos + 0.3),
                        arrowstyle='->', mutation_scale=20, 
                        color=color_arrow, linewidth=2)
ax.add_patch(arrow1)

# CNN Backbone Section
y_pos = 7.5
ax.text(7, y_pos + 1.2, 'CNN Backbone', ha='center', va='center', 
        fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', 
        facecolor=color_cnn, alpha=0.3))

# Conv Block 1
conv1_box = FancyBboxPatch((3.3, y_pos), 2.2, 0.8, 
                           boxstyle="round,pad=0.05", 
                           facecolor=color_cnn, 
                           edgecolor='black', linewidth=1.5)
ax.add_patch(conv1_box)
ax.text(4.4, y_pos + 0.55, 'Conv Block 1', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(4.4, y_pos + 0.35, '64 filters', ha='center', va='center', fontsize=7)
ax.text(4.4, y_pos + 0.15, 'BatchNorm + ReLU', ha='center', va='center', fontsize=7)
ax.text(4.4, y_pos - 0.05, 'MaxPool (2×2)', ha='center', va='center', fontsize=7)

# Arrow
arrow2 = FancyArrowPatch((5.5, y_pos + 0.4), (6.0, y_pos + 0.4),
                        arrowstyle='->', mutation_scale=15, 
                        color=color_arrow, linewidth=1.5)
ax.add_patch(arrow2)

# Conv Block 2
conv2_box = FancyBboxPatch((6.0, y_pos), 2.2, 0.8, 
                           boxstyle="round,pad=0.05", 
                           facecolor=color_cnn, 
                           edgecolor='black', linewidth=1.5)
ax.add_patch(conv2_box)
ax.text(7.1, y_pos + 0.55, 'Conv Block 2', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(7.1, y_pos + 0.35, '128 filters', ha='center', va='center', fontsize=7)
ax.text(7.1, y_pos + 0.15, 'BatchNorm + ReLU', ha='center', va='center', fontsize=7)
ax.text(7.1, y_pos - 0.05, 'MaxPool (2×2)', ha='center', va='center', fontsize=7)

# Arrow
arrow3 = FancyArrowPatch((8.2, y_pos + 0.4), (8.7, y_pos + 0.4),
                        arrowstyle='->', mutation_scale=15, 
                        color=color_arrow, linewidth=1.5)
ax.add_patch(arrow3)

# Conv Block 3
conv3_box = FancyBboxPatch((8.7, y_pos), 2.2, 0.8, 
                           boxstyle="round,pad=0.05", 
                           facecolor=color_cnn, 
                           edgecolor='black', linewidth=1.5)
ax.add_patch(conv3_box)
ax.text(9.8, y_pos + 0.55, 'Conv Block 3', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(9.8, y_pos + 0.35, '256 filters', ha='center', va='center', fontsize=7)
ax.text(9.8, y_pos + 0.15, 'BatchNorm + ReLU', ha='center', va='center', fontsize=7)
ax.text(9.8, y_pos - 0.05, '56×56×256', ha='center', va='center', fontsize=7)

# Arrow down to patch embedding
arrow4 = FancyArrowPatch((9.8, y_pos), (9.8, y_pos - 0.5),
                        arrowstyle='->', mutation_scale=20, 
                        color=color_arrow, linewidth=2)
ax.add_patch(arrow4)

# Vision Transformer Section
y_pos = 6.0
ax.text(7, y_pos + 1.0, 'Vision Transformer', ha='center', va='center', 
        fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', 
        facecolor=color_vit, alpha=0.3))

# Patch Embedding
patch_box = FancyBboxPatch((8.3, y_pos), 3.0, 0.6, 
                           boxstyle="round,pad=0.05", 
                           facecolor=color_vit, 
                           edgecolor='black', linewidth=1.5)
ax.add_patch(patch_box)
ax.text(9.8, y_pos + 0.4, 'Patch Embedding', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(9.8, y_pos + 0.1, 'Conv 16×16, stride=16 → 256-dim', ha='center', va='center', fontsize=7)

# Arrow
arrow5 = FancyArrowPatch((9.8, y_pos), (9.8, y_pos - 0.5),
                        arrowstyle='->', mutation_scale=15, 
                        color=color_arrow, linewidth=1.5)
ax.add_patch(arrow5)

# Positional Encoding + CLS Token
y_pos = 5.0
pos_box = FancyBboxPatch((8.3, y_pos), 3.0, 0.6, 
                         boxstyle="round,pad=0.05", 
                         facecolor=color_vit, 
                         edgecolor='black', linewidth=1.5)
ax.add_patch(pos_box)
ax.text(9.8, y_pos + 0.4, 'Positional Encoding', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(9.8, y_pos + 0.1, '+ CLS Token', ha='center', va='center', fontsize=7)

# Arrow
arrow6 = FancyArrowPatch((9.8, y_pos), (9.8, y_pos - 0.5),
                        arrowstyle='->', mutation_scale=15, 
                        color=color_arrow, linewidth=1.5)
ax.add_patch(arrow6)

# Transformer Encoder
y_pos = 3.5
trans_box = FancyBboxPatch((8.0, y_pos), 3.6, 1.0, 
                           boxstyle="round,pad=0.05", 
                           facecolor=color_vit, 
                           edgecolor='black', linewidth=2)
ax.add_patch(trans_box)
ax.text(9.8, y_pos + 0.7, 'Transformer Encoder', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(9.8, y_pos + 0.45, '6 Layers', ha='center', va='center', fontsize=8)
ax.text(9.8, y_pos + 0.25, '8 Attention Heads', ha='center', va='center', fontsize=8)
ax.text(9.8, y_pos + 0.05, '256-dim embeddings', ha='center', va='center', fontsize=8)

# Arrow
arrow7 = FancyArrowPatch((9.8, y_pos), (9.8, y_pos - 0.5),
                        arrowstyle='->', mutation_scale=20, 
                        color=color_arrow, linewidth=2)
ax.add_patch(arrow7)

# Classification Head
y_pos = 2.3
norm_box = FancyBboxPatch((8.5, y_pos), 2.6, 0.4, 
                          boxstyle="round,pad=0.05", 
                          facecolor=color_output, 
                          edgecolor='black', linewidth=1.5)
ax.add_patch(norm_box)
ax.text(9.8, y_pos + 0.2, 'Layer Normalization', ha='center', va='center', fontsize=9, fontweight='bold')

# Arrow
arrow8 = FancyArrowPatch((9.8, y_pos), (9.8, y_pos - 0.4),
                        arrowstyle='->', mutation_scale=15, 
                        color=color_arrow, linewidth=1.5)
ax.add_patch(arrow8)

# FC Layers
y_pos = 1.4
fc_box = FancyBboxPatch((8.3, y_pos), 3.0, 0.5, 
                        boxstyle="round,pad=0.05", 
                        facecolor=color_output, 
                        edgecolor='black', linewidth=1.5)
ax.add_patch(fc_box)
ax.text(9.8, y_pos + 0.3, 'FC (256→128) + ReLU + Dropout', ha='center', va='center', fontsize=8)
ax.text(9.8, y_pos + 0.05, 'FC (128→138)', ha='center', va='center', fontsize=8)

# Arrow
arrow9 = FancyArrowPatch((9.8, y_pos), (9.8, y_pos - 0.4),
                        arrowstyle='->', mutation_scale=20, 
                        color=color_arrow, linewidth=2)
ax.add_patch(arrow9)

# Output
y_pos = 0.5
output_box = FancyBboxPatch((8.5, y_pos), 2.6, 0.5, 
                            boxstyle="round,pad=0.05", 
                            facecolor=color_output, 
                            edgecolor='black', linewidth=2)
ax.add_patch(output_box)
ax.text(9.8, y_pos + 0.35, 'Output', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(9.8, y_pos + 0.1, '138 Person Classes', ha='center', va='center', fontsize=8)

# Add side annotations with architecture details
y_info = 7.5
ax.text(0.3, y_info, 'Architecture Details:', ha='left', va='top', 
        fontsize=9, fontweight='bold')
ax.text(0.3, y_info - 0.4, '• Input: 224×224 grayscale', ha='left', va='top', fontsize=7)
ax.text(0.3, y_info - 0.7, '• CNN: Local features', ha='left', va='top', fontsize=7)
ax.text(0.3, y_info - 1.0, '• ViT: Global context', ha='left', va='top', fontsize=7)
ax.text(0.3, y_info - 1.3, '• Patch size: 16×16', ha='left', va='top', fontsize=7)
ax.text(0.3, y_info - 1.6, '• Dropout: 0.1', ha='left', va='top', fontsize=7)

y_info = 5.0
ax.text(0.3, y_info, 'Training Details:', ha='left', va='top', 
        fontsize=9, fontweight='bold')
ax.text(0.3, y_info - 0.4, '• Optimizer: AdamW', ha='left', va='top', fontsize=7)
ax.text(0.3, y_info - 0.7, '• LR: 1e-4', ha='left', va='top', fontsize=7)
ax.text(0.3, y_info - 1.0, '• Scheduler: Cosine', ha='left', va='top', fontsize=7)
ax.text(0.3, y_info - 1.3, '• Batch size: 16', ha='left', va='top', fontsize=7)
ax.text(0.3, y_info - 1.6, '• Epochs: 50', ha='left', va='top', fontsize=7)

y_info = 2.5
ax.text(0.3, y_info, 'Data Augmentation:', ha='left', va='top', 
        fontsize=9, fontweight='bold')
ax.text(0.3, y_info - 0.4, '• Random rotation ±10°', ha='left', va='top', fontsize=7)
ax.text(0.3, y_info - 0.7, '• Random translation ±10%', ha='left', va='top', fontsize=7)
ax.text(0.3, y_info - 1.0, '• Normalization', ha='left', va='top', fontsize=7)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=color_input, edgecolor='black', label='Input'),
    mpatches.Patch(facecolor=color_cnn, edgecolor='black', label='CNN Layers'),
    mpatches.Patch(facecolor=color_vit, edgecolor='black', label='ViT Components'),
    mpatches.Patch(facecolor=color_output, edgecolor='black', label='Classification')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=8, 
          frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('cnn_vit_architecture_diagram.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Diagram saved as: cnn_vit_architecture_diagram.png")
plt.close()
