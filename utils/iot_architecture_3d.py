"""
3D IoT Architecture Visualization
Creates interactive 3D visualizations of IoT systems
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

class IoTArchitecture3D:
    def __init__(self):
        self.layers = self._define_layers()
        self.components = self._define_components()
        self.threats = self._define_threats()
        print("IoTArchitecture3D initialized - Ready for 3D visualization")
    
    def _define_layers(self):
        """Define IoT architecture layers"""
        return {
            'Perception': {
                'height': 1,
                'color': '#3498db',
                'description': 'Sensors and actuators that interact with physical world',
                'security_level': 3  # 1-10 scale
            },
            'Network': {
                'height': 2,
                'color': '#2ecc71',
                'description': 'Communication protocols and network infrastructure',
                'security_level': 5
            },
            'Application': {
                'height': 3,
                'color': '#e74c3c',
                'description': 'Services, applications, and data processing',
                'security_level': 7
            },
            'Cloud': {
                'height': 4,
                'color': '#f39c12',
                'description': 'Cloud platforms and remote services',
                'security_level': 6
            }
        }
    
    def _define_components(self):
        """Define components in each layer"""
        return {
            'Perception': [
                {'name': 'Temperature Sensor', 'x': 1, 'y': 1, 'size': 0.5, 'color': '#2980b9'},
                {'name': 'Motion Detector', 'x': 3, 'y': 1, 'size': 0.6, 'color': '#2980b9'},
                {'name': 'Smart Camera', 'x': 5, 'y': 1, 'size': 0.8, 'color': '#2980b9'},
                {'name': 'GPS Module', 'x': 7, 'y': 1, 'size': 0.4, 'color': '#2980b9'},
                {'name': 'RFID Reader', 'x': 9, 'y': 1, 'size': 0.5, 'color': '#2980b9'}
            ],
            'Network': [
                {'name': 'Wi-Fi Router', 'x': 2, 'y': 2, 'size': 0.7, 'color': '#27ae60'},
                {'name': 'Zigbee Gateway', 'x': 4, 'y': 2, 'size': 0.6, 'color': '#27ae60'},
                {'name': '5G Module', 'x': 6, 'y': 2, 'size': 0.5, 'color': '#27ae60'},
                {'name': 'Bluetooth Hub', 'x': 8, 'y': 2, 'size': 0.4, 'color': '#27ae60'}
            ],
            'Application': [
                {'name': 'Mobile App', 'x': 3, 'y': 3, 'size': 0.8, 'color': '#c0392b'},
                {'name': 'Web Dashboard', 'x': 5, 'y': 3, 'size': 1.0, 'color': '#c0392b'},
                {'name': 'Analytics Engine', 'x': 7, 'y': 3, 'size': 0.9, 'color': '#c0392b'}
            ],
            'Cloud': [
                {'name': 'AWS IoT Core', 'x': 4, 'y': 4, 'size': 1.2, 'color': '#d35400'},
                {'name': 'Azure IoT Hub', 'x': 6, 'y': 4, 'size': 1.1, 'color': '#d35400'}
            ]
        }
    
    def _define_threats(self):
        """Define security threats for each layer"""
        return {
            'Perception': [
                {'name': 'Physical Tampering', 'risk': 'High', 'x': 2, 'y': 0.5},
                {'name': 'Sensor Spoofing', 'risk': 'Medium', 'x': 4, 'y': 0.5},
                {'name': 'Battery Drain', 'risk': 'Medium', 'x': 6, 'y': 0.5},
                {'name': 'Eavesdropping', 'risk': 'High', 'x': 8, 'y': 0.5}
            ],
            'Network': [
                {'name': 'DoS Attacks', 'risk': 'Critical', 'x': 3, 'y': 1.5},
                {'name': 'MITM Attacks', 'risk': 'High', 'x': 5, 'y': 1.5},
                {'name': 'Protocol Exploits', 'risk': 'Medium', 'x': 7, 'y': 1.5}
            ],
            'Application': [
                {'name': 'SQL Injection', 'risk': 'High', 'x': 4, 'y': 2.5},
                {'name': 'XSS Attacks', 'risk': 'Medium', 'x': 6, 'y': 2.5},
                {'name': 'API Abuse', 'risk': 'High', 'x': 5, 'y': 2.5}
            ],
            'Cloud': [
                {'name': 'Data Breach', 'risk': 'Critical', 'x': 5, 'y': 3.5},
                {'name': 'DDoS Attacks', 'risk': 'High', 'x': 5.5, 'y': 3.5},
                {'name': 'Account Hijack', 'risk': 'Medium', 'x': 4.5, 'y': 3.5}
            ]
        }
    
    def visualize_architecture_3d(self):
        """Create 3D visualization of IoT architecture"""
        print("Creating 3D IoT Architecture Visualization...")
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw layers as planes
        for layer_idx, (layer_name, layer_info) in enumerate(self.layers.items()):
            z = layer_idx  # Layer height
            
            # Create layer plane
            x = np.array([0, 10])
            y = np.array([0, 5])
            X, Y = np.meshgrid(x, y)
            Z = np.full_like(X, z)
            
            # Plot layer plane with transparency
            ax.plot_surface(X, Y, Z, alpha=0.1, color=layer_info['color'],
                           label=f'{layer_name} Layer')
            
            # Add layer label
            ax.text(5, 2.5, z + 0.1, layer_name, color=layer_info['color'],
                   fontsize=14, fontweight='bold', ha='center')
            
            # Add components in this layer
            if layer_name in self.components:
                for component in self.components[layer_name]:
                    # Plot component as a sphere
                    u = np.linspace(0, 2 * np.pi, 20)
                    v = np.linspace(0, np.pi, 20)
                    x_sphere = component['size'] * np.outer(np.cos(u), np.sin(v)) + component['x']
                    y_sphere = component['size'] * np.outer(np.sin(u), np.sin(v)) + component['y']
                    z_sphere = component['size'] * np.outer(np.ones(np.size(u)), np.cos(v)) + z
                    
                    ax.plot_surface(x_sphere, y_sphere, z_sphere,
                                   color=component['color'], alpha=0.7)
                    
                    # Add component label
                    ax.text(component['x'], component['y'], z + 0.3,
                           component['name'], fontsize=8, ha='center')
        
        # Add threats as red warning symbols
        for layer_name, threats in self.threats.items():
            layer_idx = list(self.layers.keys()).index(layer_name)
            z = layer_idx - 0.2  # Slightly below the layer
            
            for threat in threats:
                # Plot threat as red X
                ax.scatter(threat['x'], threat['y'], z,
                          color='red', marker='x', s=100, linewidths=2)
                
                # Add threat label with risk level
                risk_color = {
                    'Critical': '#e74c3c',
                    'High': '#f39c12',
                    'Medium': '#f1c40f',
                    'Low': '#2ecc71'
                }.get(threat['risk'], '#95a5a6')
                
                ax.text(threat['x'], threat['y'] - 0.3, z,
                       f"{threat['name']}\n({threat['risk']})",
                       fontsize=7, color=risk_color, ha='center')
        
        # Add connections between components (simplified)
        print("  Adding connections between components...")
        
        # Connect sensors to gateway
        ax.plot([1, 2], [1, 2], [0, 1], 'b-', alpha=0.3, linewidth=1)  # Temp sensor to Wi-Fi
        ax.plot([3, 2], [1, 2], [0, 1], 'b-', alpha=0.3, linewidth=1)  # Motion to Wi-Fi
        ax.plot([5, 2], [1, 2], [0, 1], 'b-', alpha=0.3, linewidth=1)  # Camera to Wi-Fi
        
        # Connect gateway to cloud
        ax.plot([2, 4], [2, 4], [1, 3], 'g-', alpha=0.3, linewidth=1)  # Wi-Fi to AWS
        ax.plot([2, 6], [2, 4], [1, 3], 'g-', alpha=0.3, linewidth=1)  # Wi-Fi to Azure
        
        # Connect cloud to applications
        ax.plot([4, 5], [4, 3], [3, 2], 'r-', alpha=0.3, linewidth=1)  # AWS to Dashboard
        ax.plot([6, 5], [4, 3], [3, 2], 'r-', alpha=0.3, linewidth=1)  # Azure to Dashboard
        
        # Set labels and title
        ax.set_xlabel('X Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=12, fontweight='bold')
        ax.set_zlabel('Architecture Layer', fontsize=12, fontweight='bold')
        ax.set_title('3D IoT System Architecture with Security Threats', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Set viewing angle
        ax.view_init(elev=25, azim=45)
        
        # Create legend
        print("  Creating legend...")
        legend_elements = [
            mpatches.Patch(color='#3498db', alpha=0.3, label='Perception Layer'),
            mpatches.Patch(color='#2ecc71', alpha=0.3, label='Network Layer'),
            mpatches.Patch(color='#e74c3c', alpha=0.3, label='Application Layer'),
            mpatches.Patch(color='#f39c12', alpha=0.3, label='Cloud Layer'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980b9',
                      markersize=10, label='Sensor Devices'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60',
                      markersize=10, label='Network Devices'),
            plt.Line2D([0], [0], marker='x', color='red', markersize=10,
                      label='Security Threats', linewidth=2)
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        
        # Save the figure
        import os
        os.makedirs("outputs/graphs", exist_ok=True)
        save_path = "outputs/graphs/iot_architecture_3d.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 3D Architecture visualization saved to: {save_path}")
        return save_path
    
    def generate_comprehensive_report(self):
        """Generate comprehensive architecture report"""
        print("Generating Comprehensive IoT Architecture Report...")
        
        report = []
        report.append("="*70)
        report.append("IoT SYSTEM ARCHITECTURE ANALYSIS REPORT")
        report.append("="*70)
        
        report.append("\n1. ARCHITECTURE OVERVIEW:")
        report.append("-"*40)
        
        total_components = 0
        for layer_name, layer_info in self.layers.items():
            num_components = len(self.components.get(layer_name, []))
            total_components += num_components
            
            report.append(f"\n{layer_name.upper()} LAYER:")
            report.append(f"  • Description: {layer_info['description']}")
            report.append(f"  • Security Level: {layer_info['security_level']}/10")
            report.append(f"  • Components: {num_components} devices/services")
            
            if layer_name in self.components:
                report.append(f"  • Key Components:")
                for comp in self.components[layer_name][:3]:  # Show first 3
                    report.append(f"    - {comp['name']}")
                if num_components > 3:
                    report.append(f"    - ... and {num_components-3} more")
        
        report.append(f"\nTotal System Components: {total_components}")
        
        report.append("\n\n2. SECURITY THREAT ANALYSIS:")
        report.append("-"*40)
        
        for layer_name, threats in self.threats.items():
            report.append(f"\n{layer_name.upper()} LAYER THREATS:")
            for threat in threats:
                report.append(f"  • {threat['name']} (Risk: {threat['risk']})")
        
        report.append("\n\n3. SECURITY RECOMMENDATIONS:")
        report.append("-"*40)
        
        recommendations = {
            'Perception': [
                "Implement physical security measures",
                "Use tamper-resistant hardware",
                "Encrypt sensor data",
                "Implement device authentication"
            ],
            'Network': [
                "Use strong encryption (TLS 1.3+)",
                "Implement network segmentation",
                "Deploy intrusion detection systems",
                "Use VPNs for sensitive communications"
            ],
            'Application': [
                "Implement strong access controls",
                "Regular security audits and updates",
                "Use API gateways with rate limiting",
                "Deploy AI-based anomaly detection"
            ],
            'Cloud': [
                "Enable multi-factor authentication",
                "Implement data encryption at rest",
                "Use cloud security monitoring",
                "Regular backup and disaster recovery"
            ]
        }
        
        for layer, recs in recommendations.items():
            report.append(f"\n{layer.upper()} LAYER:")
            for rec in recs:
                report.append(f"  • {rec}")
        
        report.append("\n\n4. SYSTEM SECURITY SCORE:")
        report.append("-"*40)
        
        # Calculate security score
        base_scores = [layer['security_level'] for layer in self.layers.values()]
        avg_security = sum(base_scores) / len(base_scores)
        
        # Adjust for threats
        threat_penalty = 0
        for threats in self.threats.values():
            for threat in threats:
                risk_penalty = {
                    'Critical': 2.0,
                    'High': 1.5,
                    'Medium': 1.0,
                    'Low': 0.5
                }.get(threat['risk'], 0)
                threat_penalty += risk_penalty
        
        final_score = max(1, min(10, avg_security - threat_penalty/len(self.threats)))
        
        report.append(f"  • Base Security: {avg_security:.1f}/10")
        report.append(f"  • Threat Penalty: -{threat_penalty/len(self.threats):.1f}")
        report.append(f"  • Final Security Score: {final_score:.1f}/10")
        
        # Security rating
        if final_score >= 8:
            rating = "EXCELLENT"
            color = "🟢"
        elif final_score >= 6:
            rating = "GOOD"
            color = "🟡"
        elif final_score >= 4:
            rating = "FAIR"
            color = "🟠"
        else:
            rating = "POOR"
            color = "🔴"
        
        report.append(f"  • Security Rating: {color} {rating}")
        
        report.append("\n\n5. MITIGATION STRATEGY:")
        report.append("-"*40)
        report.append("  • Short-term (1-3 months):")
        report.append("    - Deploy basic intrusion detection")
        report.append("    - Implement network segmentation")
        report.append("    - Update all device firmware")
        
        report.append("\n  • Medium-term (3-6 months):")
        report.append("    - Deploy AI-based anomaly detection")
        report.append("    - Implement comprehensive monitoring")
        report.append("    - Conduct penetration testing")
        
        report.append("\n  • Long-term (6-12 months):")
        report.append("    - Implement zero-trust architecture")
        report.append("    - Deploy blockchain for data integrity")
        report.append("    - Establish incident response team")
        
        report.append("\n" + "="*70)
        report.append("END OF REPORT")
        report.append("="*70)
        
        # Save report to file
        import os
        os.makedirs("outputs/reports", exist_ok=True)
        report_path = "outputs/reports/iot_architecture_report.txt"
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Comprehensive report saved to: {report_path}")
        
        # Print summary to console
        print("\n📋 REPORT SUMMARY:")
        print(f"   • Architecture Layers: {len(self.layers)}")
        print(f"   • Total Components: {total_components}")
        print(f"   • Identified Threats: {sum(len(t) for t in self.threats.values())}")
        print(f"   • Security Score: {final_score:.1f}/10 ({rating})")
        
        return '\n'.join(report[:50])  # Return first 50 lines for display
    
    def get_detailed_architecture(self):
        """Get detailed architecture information"""
        return {
            'layers': self.layers,
            'components': self.components,
            'threats': self.threats
        }
    
    def analyze_threat_landscape(self):
        """Analyze threat landscape across all layers"""
        threat_analysis = {}
        
        for layer_name, threats in self.threats.items():
            threat_counts = {
                'Critical': 0,
                'High': 0,
                'Medium': 0,
                'Low': 0
            }
            
            for threat in threats:
                threat_counts[threat['risk']] += 1
            
            total_threats = len(threats)
            risk_score = (
                threat_counts['Critical'] * 10 +
                threat_counts['High'] * 7 +
                threat_counts['Medium'] * 4 +
                threat_counts['Low'] * 1
            ) / max(1, total_threats)
            
            threat_analysis[layer_name] = {
                'total_threats': total_threats,
                'threat_distribution': threat_counts,
                'risk_score': min(10, risk_score),
                'impact': 'Critical' if threat_counts['Critical'] > 0 else 
                         'High' if threat_counts['High'] > 0 else
                         'Medium' if threat_counts['Medium'] > 0 else 'Low',
                'detection_difficulty': 'High' if layer_name in ['Network', 'Cloud'] else 'Medium'
            }
        
        return threat_analysis

# Test the 3D visualizer
if __name__ == "__main__":
    print("Testing IoTArchitecture3D...")
    
    iot_3d = IoTArchitecture3D()
    
    # Generate 3D visualization
    viz_path = iot_3d.visualize_architecture_3d()
    
    # Generate report
    report = iot_3d.generate_comprehensive_report()
    
    # Analyze threats
    threat_analysis = iot_3d.analyze_threat_landscape()
    
    print("\n📊 THREAT ANALYSIS SUMMARY:")
    for layer, analysis in threat_analysis.items():
        print(f"\n{layer.upper()}:")
        print(f"  • Threats: {analysis['total_threats']}")
        print(f"  • Risk Score: {analysis['risk_score']:.1f}/10")
        print(f"  • Impact: {analysis['impact']}")
        print(f"  • Detection: {analysis['detection_difficulty']}")
    
    print("\n✅ IoTArchitecture3D test complete!")