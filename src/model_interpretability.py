"""
Model Interpretability Module using SHAP and LIME

This module helps explain why the model makes certain predictions
by showing which features are most important for each decision.
"""

import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')


class ModelInterpreter:
    """Class to generate SHAP and LIME explanations for model predictions"""
    
    def __init__(self, model, X_train, feature_names):
        """Initialize with model, training data, and feature names"""
        self.model = model
        self.X_train = X_train if isinstance(X_train, np.ndarray) else X_train.values
        self.feature_names = feature_names
        
        # Try to create SHAP explainer (TreeExplainer works best for tree models)
        try:
            self.shap_explainer = shap.TreeExplainer(model)
        except:
            try:
                # Fallback to KernelExplainer if TreeExplainer doesn't work
                self.shap_explainer = shap.KernelExplainer(
                    model.predict_proba, 
                    shap.sample(self.X_train, 100)
                )
            except:
                self.shap_explainer = None
        
        # Setup LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train,
            feature_names=self.feature_names,
            class_names=['No Diabetes', 'Diabetes'],
            mode='classification',
            discretize_continuous=True
        )
    
    def get_shap_values(self, X_instance):
        """Calculate SHAP values to show feature contributions"""
        if self.shap_explainer is None:
            return None
        
        # Make sure input is in correct format
        if len(X_instance.shape) == 1:
            X_instance = X_instance.reshape(1, -1)
        
        try:
            shap_values = self.shap_explainer.shap_values(X_instance)
            
            # Handle binary classification case
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Get values for diabetes class
            
            # Get baseline value
            if hasattr(self.shap_explainer, 'expected_value'):
                base_value = self.shap_explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[1]
            else:
                base_value = 0.5
            
            return {
                'shap_values': shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                'base_value': float(base_value),
                'feature_names': self.feature_names
            }
        except Exception as e:
            print(f"SHAP calculation error: {e}")
            return None
    
    def get_lime_explanation(self, X_instance, num_features=8):
        """Generate LIME explanation showing top feature contributions"""
        try:
            # Make sure input is 1D
            if len(X_instance.shape) > 1:
                X_instance = X_instance.flatten()
            
            # Generate explanation
            explanation = self.lime_explainer.explain_instance(
                X_instance,
                self.model.predict_proba,
                num_features=num_features
            )
            
            feature_contributions = explanation.as_list()
            predict_proba = explanation.predict_proba
            
            return {
                'feature_contributions': feature_contributions,
                'predict_proba': predict_proba.tolist() if hasattr(predict_proba, 'tolist') else predict_proba,
                'local_pred': explanation.local_pred[0] if hasattr(explanation, 'local_pred') else None
            }
        except Exception as e:
            print(f"LIME calculation error: {e}")
            return None
    
    def plot_shap_waterfall(self, shap_data, X_instance):
        """Create waterfall plot showing SHAP feature contributions"""
        if shap_data is None:
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            shap_vals = shap_data['shap_values']
            feature_names = shap_data['feature_names']
            base_value = shap_data['base_value']
            
            if len(X_instance.shape) > 1:
                X_instance = X_instance.flatten()
            
            # Sort features by importance
            indices = np.argsort(np.abs(shap_vals))[::-1]
            
            cumsum = base_value
            y_pos = np.arange(len(feature_names))
            
            # Red for positive impact, blue for negative
            colors = ['#ff0051' if val > 0 else '#008bfb' for val in shap_vals[indices]]
            ax.barh(y_pos, shap_vals[indices], color=colors, alpha=0.7)
            
            # Add feature names with values
            labels = [f"{feature_names[i]} = {X_instance[i]:.2f}" for i in indices]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel('SHAP Value (impact on prediction)', fontsize=12)
            ax.set_title('Feature Contributions to Prediction', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
            
            # Add base value annotation
            ax.text(0.02, 0.98, f'Base value: {base_value:.3f}', 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return image_base64
        except Exception as e:
            print(f"SHAP plot error: {e}")
            return None
    
    def plot_lime_explanation(self, lime_data):
        """Create bar plot for LIME feature importance"""
        if lime_data is None:
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract features and values
            feature_contributions = lime_data['feature_contributions']
            features = [item[0] for item in feature_contributions]
            values = [item[1] for item in feature_contributions]
            
            # Create horizontal bar plot
            colors = ['#ff0051' if val > 0 else '#008bfb' for val in values]
            y_pos = np.arange(len(features))
            
            ax.barh(y_pos, values, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('Feature Contribution', fontsize=12)
            ax.set_title('LIME Feature Importance (Local Explanation)', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return image_base64
        except Exception as e:
            print(f"LIME plot error: {e}")
            return None


def explain_prediction(model, X_train, X_instance, feature_names):
    """Generate SHAP and LIME explanations for a prediction"""
    # Initialize interpreter
    interpreter = ModelInterpreter(model, X_train, feature_names)
    
    # Get SHAP explanation
    shap_data = interpreter.get_shap_values(X_instance)
    shap_plot = interpreter.plot_shap_waterfall(shap_data, X_instance) if shap_data else None
    
    # Get LIME explanation
    lime_data = interpreter.get_lime_explanation(X_instance)
    lime_plot = interpreter.plot_lime_explanation(lime_data) if lime_data else None
    
    # Prepare human-readable explanation
    explanation_text = generate_explanation_text(shap_data, lime_data, feature_names, X_instance)
    
    return {
        'shap_data': shap_data,
        'shap_plot': shap_plot,
        'lime_data': lime_data,
        'lime_plot': lime_plot,
        'explanation_text': explanation_text
    }


def generate_explanation_text(shap_data, lime_data, feature_names, X_instance):
    """Convert SHAP/LIME data into readable text explanation"""
    if shap_data is None and lime_data is None:
        return "Unable to generate explanation. Please try again."
    
    explanation = []
    
    # Use SHAP data if available
    if shap_data:
        shap_vals = shap_data['shap_values']
        if len(X_instance.shape) > 1:
            X_instance = X_instance.flatten()
        
        # Get top 3 positive and negative contributors
        indices = np.argsort(np.abs(shap_vals))[::-1][:3]
        
        explanation.append("**Key Factors Influencing This Prediction:**\n")
        for idx in indices:
            feature = feature_names[idx]
            value = X_instance[idx]
            contribution = shap_vals[idx]
            
            direction = "increases" if contribution > 0 else "decreases"
            explanation.append(
                f"- **{feature}** (value: {value:.2f}): {direction} diabetes risk by {abs(contribution):.3f}"
            )
    
    # Add LIME insights if available
    if lime_data:
        explanation.append("\n**Alternative View (LIME):**\n")
        for feature_desc, contribution in lime_data['feature_contributions'][:3]:
            direction = "supports" if contribution > 0 else "opposes"
            explanation.append(f"- {feature_desc}: {direction} diabetes prediction")
    
    return "\n".join(explanation)
