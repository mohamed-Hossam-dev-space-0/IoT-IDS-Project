class IoTArchitecture:
    def __init__(self):
        """Initialize IoT architecture components"""
        self.layers = self._define_layers()
        self.attack_points = self._define_attack_points()
        self.protocols = self._define_protocols()
    
    def _define_layers(self):
        """Define IoT architecture layers"""
        return {
            'Perception Layer': [
                'Sensors (Temperature, Motion, Light)',
                'Actuators (Motors, Switches, Valves)',
                'RFID Tags & Readers',
                'Camera Modules',
                'GPS Modules'
            ],
            'Network Layer': [
                'Gateways/Routers',
                'Wireless Access Points',
                'Network Switches',
                'Communication Protocols',
                'Edge Computing Nodes'
            ],
            'Application Layer': [
                'Cloud Servers',
                'Mobile Applications',
                'Web Dashboards',
                'Data Analytics Platform',
                'User Management System'
            ]
        }
    
    def _define_attack_points(self):
        """Define potential attack points in each layer"""
        return {
            'Perception Layer': [
                'Physical tampering of devices',
                'Eavesdropping on sensor communications',
                'Sensor data spoofing/injection',
                'Battery exhaustion attacks',
                'Jamming wireless signals'
            ],
            'Network Layer': [
                'Man-in-the-Middle (MITM) attacks',
                'Denial of Service (DoS/DDoS)',
                'Routing attacks (spoofing, sinkhole)',
                'Protocol vulnerability exploitation',
                'Unauthorized access to network'
            ],
            'Application Layer': [
                'SQL injection attacks',
                'Cross-site scripting (XSS)',
                'Authentication bypass',
                'Data leakage/exfiltration',
                'API abuse/exploitation'
            ]
        }
    
    def _define_protocols(self):
        """Define IoT communication protocols"""
        return {
            'Short Range': ['Bluetooth', 'ZigBee', 'Z-Wave', 'Wi-Fi'],
            'Long Range': ['LoRaWAN', 'Sigfox', 'NB-IoT', 'LTE-M'],
            'IP-Based': ['MQTT', 'CoAP', 'HTTP/HTTPS', 'WebSocket']
        }
    
    def get_architecture(self):
        """Get complete IoT architecture"""
        return self.layers
    
    def get_attack_points(self):
        """Get attack points for each layer"""
        return self.attack_points
    
    def get_protocols(self):
        """Get IoT communication protocols"""
        return self.protocols
    
    def analyze_vulnerabilities(self, layer):
        """
        Analyze vulnerabilities for a specific layer
        
        Args:
            layer: Layer name ('Perception', 'Network', 'Application')
        
        Returns:
            Dictionary of vulnerabilities
        """
        vulnerabilities = {
            'Perception Layer': {
                'Resource Constraints': 'Limited computation, memory, and power',
                'Physical Exposure': 'Devices often in unattended locations',
                'Lack of Encryption': 'Many sensors use plaintext communication',
                'Weak Authentication': 'Default or no authentication mechanisms'
            },
            'Network Layer': {
                'Wireless Interception': 'Radio signals can be easily intercepted',
                'Protocol Flaws': 'Many IoT protocols lack security features',
                'DoS Vulnerabilities': 'Limited bandwidth and processing power',
                'Spoofing Attacks': 'Easy to spoof device identities'
            },
            'Application Layer': {
                'Insecure APIs': 'Poorly implemented REST/Web APIs',
                'Weak Credentials': 'Default passwords and weak authentication',
                'Data Privacy': 'Sensitive data exposure in transit/storage',
                'Update Mechanisms': 'Lack of secure update processes'
            }
        }
        
        return vulnerabilities.get(layer, {})
    
    def get_security_recommendations(self):
        """
        Get security recommendations for each layer
        """
        return {
            'Perception Layer': [
                'Use tamper-resistant hardware',
                'Implement device authentication',
                'Encrypt sensor data',
                'Deploy physical security measures'
            ],
            'Network Layer': [
                'Use VPNs for sensitive communications',
                'Implement intrusion detection systems',
                'Use strong encryption (TLS 1.3+)',
                'Deploy network segmentation'
            ],
            'Application Layer': [
                'Implement strong access controls',
                'Regular security audits and updates',
                'Use API gateways with rate limiting',
                'Deploy AI-based anomaly detection'
            ]
        }
    
    def generate_architecture_report(self):
        """
        Generate comprehensive architecture report
        """
        report = []
        report.append("="*60)
        report.append("IoT SYSTEM ARCHITECTURE ANALYSIS REPORT")
        report.append("="*60)
        
        report.append("\n1. ARCHITECTURE LAYERS:")
        report.append("-"*40)
        for layer, components in self.layers.items():
            report.append(f"\n{layer}:")
            for component in components:
                report.append(f"  • {component}")
        
        report.append("\n\n2. COMMUNICATION PROTOCOLS:")
        report.append("-"*40)
        for category, protocols in self.protocols.items():
            report.append(f"\n{category}:")
            for protocol in protocols:
                report.append(f"  • {protocol}")
        
        report.append("\n\n3. SECURITY VULNERABILITIES:")
        report.append("-"*40)
        for layer in self.layers.keys():
            report.append(f"\n{layer}:")
            vulnerabilities = self.analyze_vulnerabilities(layer)
            for vuln, desc in vulnerabilities.items():
                report.append(f"  • {vuln}: {desc}")
        
        report.append("\n\n4. POTENTIAL ATTACK POINTS:")
        report.append("-"*40)
        for layer, attacks in self.attack_points.items():
            report.append(f"\n{layer}:")
            for attack in attacks:
                report.append(f"  • {attack}")
        
        report.append("\n\n5. SECURITY RECOMMENDATIONS:")
        report.append("-"*40)
        recommendations = self.get_security_recommendations()
        for layer, recs in recommendations.items():
            report.append(f"\n{layer}:")
            for rec in recs:
                report.append(f"  • {rec}")
        
        return "\n".join(report)