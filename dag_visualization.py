#!/usr/bin/env python3
"""
DAG Visualization for Synthetic Datasets
Reviewer request: Show true DAG vs. learned DAG structure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os

def compute_causal_importance(X, y, groups, feature_names):
    """Compute causal importance and correlations for visualization."""
    results = []
    
    for j, feat in enumerate(feature_names):
        # Correlation with target
        corr_y = np.corrcoef(X[:, j], y)[0, 1]
        
        # Correlation with protected
        corr_g = np.corrcoef(X[:, j], groups)[0, 1]
        
        # Partial correlation
        reg_x = LinearRegression().fit(groups.reshape(-1, 1), X[:, j])
        x_resid = X[:, j] - reg_x.predict(groups.reshape(-1, 1))
        
        reg_y = LinearRegression().fit(groups.reshape(-1, 1), y)
        y_resid = y - reg_y.predict(groups.reshape(-1, 1))
        
        partial_corr = np.corrcoef(x_resid, y_resid)[0, 1]
        
        # Causal importance
        causal_imp = abs(partial_corr) * (1 - abs(corr_g))
        
        results.append({
            'feature': feat,
            'corr_y': corr_y,
            'corr_g': corr_g,
            'partial_corr': partial_corr,
            'causal_importance': causal_imp
        })
    
    df = pd.DataFrame(results)
    df['causal_importance_norm'] = df['causal_importance'] / df['causal_importance'].max()
    
    return df


def create_dag_figure(df, feature_categories, dataset_name, threshold=0.2, output_path='dag_visualization.pdf'):
    """
    Create a figure showing:
    - True causal structure (solid lines)
    - Learned importance (bar heights)
    - Feature categories (colors)
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Feature importance with categories
    ax1 = axes[0]
    
    colors = []
    for feat in df['feature']:
        if feat in feature_categories.get('fair', []):
            colors.append('#2ECC71')  # Green for fair
        elif feat in feature_categories.get('unfair', []):
            colors.append('#E74C3C')  # Red for unfair
        else:
            colors.append('#95A5A6')  # Gray for noise
    
    bars = ax1.barh(range(len(df)), df['causal_importance_norm'], color=colors, alpha=0.8, edgecolor='black')
    
    # Add threshold line
    ax1.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold τ={threshold}')
    
    # Mark selected features
    for i, (idx, row) in enumerate(df.iterrows()):
        if row['causal_importance_norm'] >= threshold:
            ax1.text(row['causal_importance_norm'] + 0.02, i, '✓', fontsize=14, va='center', fontweight='bold')
    
    ax1.set_yticks(range(len(df)))
    ax1.set_yticklabels(df['feature'], fontsize=11)
    ax1.set_xlabel('Normalized Causal Importance', fontsize=12)
    ax1.set_title(f'{dataset_name}: Feature Selection', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1.15)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#2ECC71', label='Fair (should select)'),
        mpatches.Patch(facecolor='#E74C3C', label='Unfair (should reject)'),
        mpatches.Patch(facecolor='#95A5A6', label='Noise'),
        plt.Line2D([0], [0], color='black', linestyle='--', label=f'Threshold τ={threshold}')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Right: Causal structure diagram
    ax2 = axes[1]
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title(f'{dataset_name}: Learned Causal Structure', fontsize=14, fontweight='bold')
    
    # Draw protected attribute (center)
    protected_circle = plt.Circle((0, 0), 0.2, color='#3498DB', alpha=0.8)
    ax2.add_patch(protected_circle)
    ax2.text(0, 0, 'Protected\nAttribute', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw target (right)
    target_circle = plt.Circle((1.2, 0), 0.2, color='#9B59B6', alpha=0.8)
    ax2.add_patch(target_circle)
    ax2.text(1.2, 0, 'Target', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Draw features
    n_features = len(df)
    angles = np.linspace(np.pi/2 + 0.3, -np.pi/2 - 0.3, n_features)
    
    for i, (idx, row) in enumerate(df.iterrows()):
        feat = row['feature']
        angle = angles[i]
        x = -0.8 * np.cos(angle)
        y = 0.8 * np.sin(angle)
        
        # Color by category
        if feat in feature_categories.get('fair', []):
            color = '#2ECC71'
        elif feat in feature_categories.get('unfair', []):
            color = '#E74C3C'
        else:
            color = '#95A5A6'
        
        # Draw feature node
        feat_circle = plt.Circle((x, y), 0.12, color=color, alpha=0.8)
        ax2.add_patch(feat_circle)
        
        # Short name for display
        short_name = feat[:8] + '..' if len(feat) > 10 else feat
        ax2.text(x - 0.35, y, short_name, ha='right', va='center', fontsize=8)
        
        # Draw edge to protected (if correlated)
        if abs(row['corr_g']) > 0.1:
            line_width = abs(row['corr_g']) * 3
            ax2.annotate('', xy=(0 - 0.2, 0), xytext=(x + 0.12, y),
                        arrowprops=dict(arrowstyle='->', color='#E74C3C', 
                                       lw=line_width, alpha=0.6))
        
        # Draw edge to target (if causally important)
        if row['causal_importance_norm'] >= threshold:
            line_width = row['causal_importance_norm'] * 3
            ax2.annotate('', xy=(1.2 - 0.2, 0), xytext=(x + 0.12, y),
                        arrowprops=dict(arrowstyle='->', color='#2ECC71', 
                                       lw=line_width, alpha=0.8))
    
    # Legend for edges
    ax2.text(-1.4, -1.3, 'Edge legend:', fontsize=9, fontweight='bold')
    ax2.annotate('', xy=(-0.8, -1.3), xytext=(-1.2, -1.3),
                arrowprops=dict(arrowstyle='->', color='#2ECC71', lw=2))
    ax2.text(-0.75, -1.3, 'Causal (selected)', fontsize=8, va='center')
    
    ax2.annotate('', xy=(-0.8, -1.45), xytext=(-1.2, -1.45),
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2, alpha=0.6))
    ax2.text(-0.75, -1.45, 'Spurious (rejected)', fontsize=8, va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    
    return fig


def main():
    # Synthetic Loan
    print("="*60)
    print("SYNTHETIC LOAN - DAG VISUALIZATION")
    print("="*60)
    
    # Load or generate data
    if os.path.exists('synthetic_results/synthetic_loan_data.csv'):
        df_loan = pd.read_csv('synthetic_results/synthetic_loan_data.csv')
    elif os.path.exists('synthetic_loan_data.csv'):
        df_loan = pd.read_csv('synthetic_loan_data.csv')
    else:
        print("Generating synthetic loan data...")
        np.random.seed(42)
        n = 10000
        
        gender = np.random.binomial(1, 0.5, n)
        income = np.random.normal(50000, 15000, n)
        credit_score = np.random.normal(700, 50, n)
        employment_years = np.clip(np.random.poisson(5, n), 0, 30)
        
        works_in_tech = np.random.binomial(1, 0.15 + 0.60 * gender, n)
        has_stem_degree = np.random.binomial(1, 0.10 + 0.55 * gender, n)
        plays_golf = np.random.binomial(1, 0.05 + 0.50 * gender, n)
        
        favorite_color_blue = np.random.binomial(1, 0.3, n)
        birth_month = np.random.randint(1, 13, n)
        
        income_norm = (income - 50000) / 15000
        credit_norm = (credit_score - 700) / 50
        emp_norm = (employment_years - 5) / 3
        
        logit = (0.6 * income_norm + 0.8 * credit_norm + 0.4 * emp_norm +
                 1.2 * works_in_tech + 1.0 * has_stem_degree + 0.8 * plays_golf +
                 0.3 * np.random.randn(n))
        prob = 1 / (1 + np.exp(-logit))
        loan_approved = (np.random.rand(n) < prob).astype(int)
        
        df_loan = pd.DataFrame({
            'income': income, 'credit_score': credit_score, 
            'employment_years': employment_years,
            'works_in_tech': works_in_tech, 'has_stem_degree': has_stem_degree,
            'plays_golf': plays_golf, 'favorite_color_blue': favorite_color_blue,
            'birth_month': birth_month, 'gender': gender, 'loan_approved': loan_approved
        })
    
    feature_cols = ['income', 'credit_score', 'employment_years',
                    'works_in_tech', 'has_stem_degree', 'plays_golf',
                    'favorite_color_blue', 'birth_month']
    
    X = df_loan[feature_cols].values.astype(np.float32)
    y = df_loan['loan_approved'].values.astype(np.float32)
    groups = df_loan['gender'].values.astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compute causal importance
    imp_df = compute_causal_importance(X_scaled, y, groups, feature_cols)
    
    print("\nFeature Analysis:")
    print(imp_df[['feature', 'corr_y', 'corr_g', 'causal_importance_norm']].to_string())
    
    # Ground truth categories
    loan_categories = {
        'fair': ['income', 'credit_score', 'employment_years'],
        'unfair': ['works_in_tech', 'has_stem_degree', 'plays_golf'],
        'noise': ['favorite_color_blue', 'birth_month']
    }
    
    # Create visualization
    create_dag_figure(imp_df, loan_categories, 'Synthetic Loan', 
                     threshold=0.25, output_path='/mnt/user-data/outputs/dag_synthetic_loan.pdf')
    
    # Synthetic Hiring
    print("\n" + "="*60)
    print("SYNTHETIC HIRING - DAG VISUALIZATION")
    print("="*60)
    
    if os.path.exists('synthetic_hiring/synthetic_hiring_data.csv'):
        df_hire = pd.read_csv('synthetic_hiring/synthetic_hiring_data.csv')
    elif os.path.exists('synthetic_hiring_data.csv'):
        df_hire = pd.read_csv('synthetic_hiring_data.csv')
    else:
        print("Generating synthetic hiring data...")
        np.random.seed(42)
        n = 10000
        
        race = np.random.binomial(1, 0.6, n)
        
        years_experience = np.clip(np.random.exponential(5, n), 0, 20)
        coding_score = np.clip(50 + 20 * np.random.randn(n), 0, 100)
        education_level = np.random.choice([1, 2, 3, 4], n, p=[0.1, 0.5, 0.3, 0.1])
        portfolio_quality = np.clip(5 + 2 * np.random.randn(n), 0, 10)
        
        ivy_league = np.random.binomial(1, 0.10 + 0.30 * race, n)
        unpaid_internships = np.clip(np.random.poisson(0.5 + 1.5 * race), 0, 5)
        golf_club_member = np.random.binomial(1, 0.05 + 0.30 * race, n)
        lacrosse_player = np.random.binomial(1, 0.03 + 0.22 * race, n)
        
        birth_month = np.random.randint(1, 13, n)
        zodiac_fire_sign = np.random.binomial(1, 0.25, n)
        
        exp_norm = (years_experience - 5) / 5
        code_norm = (coding_score - 50) / 20
        edu_norm = (education_level - 2.5) / 1
        port_norm = (portfolio_quality - 5) / 2
        
        logit = (0.6 * exp_norm + 0.8 * code_norm + 0.4 * edu_norm + 0.5 * port_norm +
                 1.5 * 0.8 * ivy_league + 1.5 * 0.3 * unpaid_internships +
                 1.5 * 0.6 * golf_club_member + 1.5 * 0.5 * lacrosse_player +
                 0.3 * np.random.randn(n))
        prob = 1 / (1 + np.exp(-logit))
        hired = (np.random.rand(n) < prob).astype(int)
        
        df_hire = pd.DataFrame({
            'years_experience': years_experience, 'coding_score': coding_score,
            'education_level': education_level, 'portfolio_quality': portfolio_quality,
            'ivy_league': ivy_league, 'unpaid_internships': unpaid_internships,
            'golf_club_member': golf_club_member, 'lacrosse_player': lacrosse_player,
            'birth_month': birth_month, 'zodiac_fire_sign': zodiac_fire_sign,
            'race': race, 'hired': hired
        })
    
    feature_cols_hire = ['years_experience', 'coding_score', 'education_level', 'portfolio_quality',
                         'ivy_league', 'unpaid_internships', 'golf_club_member', 'lacrosse_player',
                         'birth_month', 'zodiac_fire_sign']
    
    X_hire = df_hire[feature_cols_hire].values.astype(np.float32)
    y_hire = df_hire['hired'].values.astype(np.float32)
    groups_hire = df_hire['race'].values.astype(int)
    
    scaler = StandardScaler()
    X_hire_scaled = scaler.fit_transform(X_hire)
    
    imp_df_hire = compute_causal_importance(X_hire_scaled, y_hire, groups_hire, feature_cols_hire)
    
    print("\nFeature Analysis:")
    print(imp_df_hire[['feature', 'corr_y', 'corr_g', 'causal_importance_norm']].to_string())
    
    hire_categories = {
        'fair': ['years_experience', 'coding_score', 'education_level', 'portfolio_quality'],
        'unfair': ['ivy_league', 'unpaid_internships', 'golf_club_member', 'lacrosse_player'],
        'noise': ['birth_month', 'zodiac_fire_sign']
    }
    
    create_dag_figure(imp_df_hire, hire_categories, 'Synthetic Hiring',
                     threshold=0.4, output_path='/mnt/user-data/outputs/dag_synthetic_hiring.pdf')
    
    print("\n✓ DAG visualizations created!")


if __name__ == '__main__':
    main()
