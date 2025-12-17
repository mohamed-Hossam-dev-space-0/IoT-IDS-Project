#!/usr/bin/env python3
"""
Advanced IoT Attack Simulation Laboratory
Run with: python simulate_attacks_real.py
"""
import numpy as np
import pandas as pd
import time
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedAttackSimulator:
    def __init__(self):
        self.attack_profiles = self._define_attack_profiles()
        self.iot_devices = self._define_iot_devices()
        self.network_state = self._initialize_network()
        print("AdvancedAttackSimulator initialized - Ready for attack simulations")
    
    def _define_attack_profiles(self):
        """Define detailed attack profiles"""
        return {
            'dos': {
                'name': 'Denial of Service',
                'description': 'Floods target with excessive requests to exhaust resources',
                'indicators': ['High packet rate', 'Small packet sizes', 'Spoofed source IPs'],
                'difficulty': 'Low',
                'impact': 'Service disruption',
                'target': 'Network bandwidth, device resources',
                'detection': 'Rate limiting, anomaly detection'
            },
            'mitm': {
                'name': 'Man-in-the-Middle',
                'description': 'Intercepts and modifies communications between two parties',
                'indicators': ['Protocol anomalies', 'Duplicate packets', 'Certificate issues'],
                'difficulty': 'Medium',
                'impact': 'Data theft, session hijacking',
                'target': 'Data confidentiality, integrity',
                'detection': 'Certificate validation, timing analysis'
            },
            'data_injection': {
                'name': 'Data Injection',
                'description': 'Injects malicious data into legitimate data streams',
                'indicators': ['Payload size anomalies', 'Data format violations', 'CRC errors'],
                'difficulty': 'High',
                'impact': 'System compromise, false readings',
                'target': 'System decisions, data integrity',
                'detection': 'Data validation, statistical analysis'
            },
            'eavesdropping': {
                'name': 'Eavesdropping',
                'description': 'Unauthorized listening to network communications',
                'indicators': ['Unencrypted traffic', 'Long session durations', 'Passive monitoring'],
                'difficulty': 'Low',
                'impact': 'Privacy violation, information leakage',
                'target': 'Data confidentiality',
                'detection': 'Encryption monitoring, traffic analysis'
            },
            'replay': {
                'name': 'Replay Attack',
                'description': 'Captures and retransmits valid data transmissions',
                'indicators': ['Duplicate packets', 'Sequence number anomalies', 'Timestamp mismatches'],
                'difficulty': 'Medium',
                'impact': 'Unauthorized actions, session hijacking',
                'target': 'Authentication systems',
                'detection': 'Timestamp validation, sequence checking'
            }
        }
    
    def _define_iot_devices(self):
        """Define typical IoT devices in a smart environment"""
        return [
            {'name': 'Smart Thermostat', 'type': 'climate', 'packets_min': 1, 'packets_max': 5, 
             'packet_size': (100, 300), 'protocol': 'MQTT'},
            {'name': 'Security Camera', 'type': 'security', 'packets_min': 10, 'packets_max': 100,
             'packet_size': (500, 5000), 'protocol': 'RTSP/HTTP'},
            {'name': 'Smart Lock', 'type': 'security', 'packets_min': 0.1, 'packets_max': 2,
             'packet_size': (50, 200), 'protocol': 'Zigbee/Bluetooth'},
            {'name': 'Motion Sensor', 'type': 'security', 'packets_min': 0.5, 'packets_max': 5,
             'packet_size': (20, 100), 'protocol': 'Z-Wave'},
            {'name': 'Smart Light', 'type': 'lighting', 'packets_min': 0.2, 'packets_max': 3,
             'packet_size': (50, 150), 'protocol': 'Wi-Fi'},
            {'name': 'Voice Assistant', 'type': 'entertainment', 'packets_min': 5, 'packets_max': 50,
             'packet_size': (500, 2000), 'protocol': 'HTTP/WebSocket'}
        ]
    
    def _initialize_network(self):
        """Initialize network state"""
        return {
            'devices': {},
            'traffic': [],
            'attacks': [],
            'start_time': datetime.now(),
            'total_packets': 0,
            'attack_packets': 0
        }
    
    def simulate_normal_traffic(self, duration_seconds=30, verbose=True):
        """Simulate normal IoT network traffic"""
        if verbose:
            print(f"\n📡 Simulating Normal IoT Traffic for {duration_seconds} seconds...")
            print("="*60)
        
        traffic_log = []
        start_time = time.time()
        packet_count = 0
        
        while time.time() - start_time < duration_seconds:
            # Randomly select a device
            device = random.choice(self.iot_devices)
            
            # Generate normal traffic pattern
            packets_per_second = random.uniform(device['packets_min'], device['packets_max'])
            packet_size = random.randint(device['packet_size'][0], device['packet_size'][1])
            
            # Create traffic entry
            traffic_entry = {
                'timestamp': datetime.now(),
                'device': device['name'],
                'packet_size': packet_size,
                'packets_per_second': packets_per_second,
                'protocol': device['protocol'],
                'source_ip': f'192.168.1.{random.randint(2, 50)}',
                'destination_ip': f'192.168.1.{random.randint(100, 200)}',
                'port': random.randint(1024, 65535),
                'is_attack': False,
                'attack_type': None
            }
            
            traffic_log.append(traffic_entry)
            packet_count += 1
            
            # Print progress
            elapsed = time.time() - start_time
            if verbose and packet_count % 10 == 0:
                print(f"  Generated {packet_count} packets | "
                      f"Elapsed: {elapsed:.1f}s/{duration_seconds}s")
            
            # Sleep to simulate real-time
            time.sleep(1 / max(1, packets_per_second))
        
        if verbose:
            print(f"\n✅ Normal traffic simulation complete!")
            print(f"   • Total packets: {packet_count}")
            print(f"   • Duration: {duration_seconds} seconds")
            print(f"   • Average rate: {packet_count/duration_seconds:.1f} packets/second")
        
        return traffic_log
    
    def simulate_dos_attack(self, duration_seconds=20, verbose=True):
        """Simulate Denial of Service attack"""
        if verbose:
            print(f"\n🚨 Simulating DoS Attack for {duration_seconds} seconds...")
            print("="*60)
        
        attack_log = []
        start_time = time.time()
        packet_count = 0
        
        # Select attack type
        attack_type = random.choice(['udp_flood', 'syn_flood', 'http_flood'])
        
        if verbose:
            print(f"Attack Type: {attack_type.replace('_', ' ').title()}")
            print(f"Target: Random IoT device in network")
            print(f"Mechanism: Packet flooding to exhaust resources")
            print(f"Expected Impact: Service degradation or outage")
        
        while time.time() - start_time < duration_seconds:
            # Generate attack packets based on attack type
            if attack_type == 'udp_flood':
                # UDP flood: many small packets
                num_packets = random.randint(100, 1000)
                packet_size = random.randint(64, 512)
                protocol = 'UDP'
                target_port = random.randint(1, 65535)
                
            elif attack_type == 'syn_flood':
                # SYN flood: TCP SYN packets
                num_packets = random.randint(50, 500)
                packet_size = random.randint(40, 60)
                protocol = 'TCP'
                target_port = 80  # Common web port
                
            else:  # http_flood
                # HTTP flood: legitimate-looking requests
                num_packets = random.randint(10, 100)
                packet_size = random.randint(500, 1500)
                protocol = 'HTTP'
                target_port = 80
        
            for _ in range(num_packets):
                attack_entry = {
                    'timestamp': datetime.now(),
                    'attack_type': f'dos_{attack_type}',
                    'packet_size': packet_size,
                    'protocol': protocol,
                    'source_ip': f'10.0.{random.randint(0, 255)}.{random.randint(0, 255)}',
                    'destination_ip': f'192.168.1.{random.randint(2, 50)}',
                    'port': target_port,
                    'is_attack': True,
                    'intensity': 'High',
                    'description': f'{attack_type} flood attack'
                }
                
                attack_log.append(attack_entry)
                packet_count += 1
            
            # Print attack progress
            elapsed = time.time() - start_time
            if verbose:
                attack_rate = packet_count / elapsed if elapsed > 0 else 0
                print(f"  ⚡ Flooding: {packet_count:,} packets sent | "
                      f"Rate: {attack_rate:,.0f} packets/sec")
            
            time.sleep(0.1)  # Small delay between bursts
        
        if verbose:
            print(f"\n✅ DoS attack simulation complete!")
            print(f"   • Total attack packets: {packet_count:,}")
            print(f"   • Attack duration: {duration_seconds} seconds")
            print(f"   • Average attack rate: {packet_count/duration_seconds:,.0f} packets/second")
            print(f"   • Attack type: {attack_type.replace('_', ' ').title()}")
            print(f"   • Detection method: Rate limiting, anomaly detection")
        
        return attack_log
    
    def simulate_mitm_attack(self, duration_seconds=25, verbose=True):
        """Simulate Man-in-the-Middle attack"""
        if verbose:
            print(f"\n👤 Simulating MITM Attack for {duration_seconds} seconds...")
            print("="*60)
        
        attack_log = []
        start_time = time.time()
        intercept_count = 0
        
        # Select MITM technique
        technique = random.choice(['arp_spoofing', 'dns_spoofing', 'ssl_stripping'])
        
        if verbose:
            print(f"MITM Technique: {technique.replace('_', ' ').title()}")
            print(f"Target: Intercepting communications between IoT devices")
            print(f"Mechanism: Position between communicating parties to intercept/modify traffic")
            print(f"Expected Impact: Data theft, session hijacking, integrity violation")
        
        while time.time() - start_time < duration_seconds:
            if technique == 'arp_spoofing':
                # ARP spoofing: fake ARP responses
                attack_entry = {
                    'timestamp': datetime.now(),
                    'attack_type': 'mitm_arp_spoofing',
                    'description': 'Sent fake ARP response mapping gateway IP to attacker MAC',
                    'source_mac': ':'.join(f'{random.randint(0, 255):02x}' for _ in range(6)),
                    'target_ip': f'192.168.1.{random.randint(2, 50)}',
                    'spoofed_ip': '192.168.1.1',  # Gateway
                    'is_attack': True,
                    'stealth_level': 'Medium',
                    'data_intercepted': random.choice(['Login credentials', 'Sensor data', 'Configuration'])
                }
                
            elif technique == 'dns_spoofing':
                # DNS spoofing: fake DNS responses
                domains = ['api.iot-cloud.com', 'update.server.com', 'config.iot.com', 'auth.device.com']
                attack_entry = {
                    'timestamp': datetime.now(),
                    'attack_type': 'mitm_dns_spoofing',
                    'description': 'Sent fake DNS response redirecting to malicious server',
                    'query_domain': random.choice(domains),
                    'spoofed_ip': f'{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}',
                    'target_device': random.choice(['voice_assistant', 'security_camera', 'thermostat']),
                    'is_attack': True,
                    'stealth_level': 'High',
                    'data_intercepted': 'DNS queries and responses'
                }
                
            else:  # ssl_stripping
                # SSL stripping: downgrade HTTPS to HTTP
                urls = ['https://login.iot.com', 'https://api.device.com', 'https://cloud.iot.com']
                attack_entry = {
                    'timestamp': datetime.now(),
                    'attack_type': 'mitm_ssl_stripping',
                    'description': 'Intercepted HTTPS request and redirected to HTTP version',
                    'original_url': random.choice(urls),
                    'stripped_url': random.choice(urls).replace('https://', 'http://'),
                    'target_device': random.choice(['voice_assistant', 'security_camera', 'thermostat']),
                    'is_attack': True,
                    'stealth_level': 'Medium',
                    'data_intercepted': 'Login credentials, session tokens'
                }
            
            attack_log.append(attack_entry)
            intercept_count += 1
            
            # Print progress
            elapsed = time.time() - start_time
            if verbose and intercept_count % 3 == 0:
                print(f"  🔍 Intercepted {intercept_count} communications | "
                      f"Elapsed: {elapsed:.1f}s/{duration_seconds}s")
                if 'data_intercepted' in attack_entry:
                    print(f"     Intercepted: {attack_entry['data_intercepted']}")
            
            time.sleep(random.uniform(2, 5))  # MITM happens intermittently
        
        if verbose:
            print(f"\n✅ MITM attack simulation complete!")
            print(f"   • Total intercepts: {intercept_count}")
            print(f"   • Attack duration: {duration_seconds} seconds")
            print(f"   • Technique: {technique.replace('_', ' ').title()}")
            print(f"   • Detection difficulty: High (requires encrypted traffic inspection)")
            print(f"   • Prevention: Use strong encryption, certificate pinning")
        
        return attack_log
    
    def simulate_data_injection_attack(self, duration_seconds=20, verbose=True):
        """Simulate Data Injection attack"""
        if verbose:
            print(f"\n💉 Simulating Data Injection Attack for {duration_seconds} seconds...")
            print("="*60)
        
        attack_log = []
        start_time = time.time()
        injection_count = 0
        
        if verbose:
            print(f"Attack Type: Data Injection")
            print(f"Target: IoT sensor data streams and control systems")
            print(f"Mechanism: Inject false readings or malicious commands into legitimate data")
            print(f"Expected Impact: System malfunctions, false alarms, wrong decisions")
        
        sensor_types = [
            ('temperature', '°C', (15, 30), 'climate control'),
            ('humidity', '%', (30, 70), 'environment monitoring'),
            ('pressure', 'hPa', (900, 1100), 'weather monitoring'),
            ('motion', 'detected/not', (0, 1), 'security system'),
            ('door_status', 'open/closed', (0, 1), 'access control'),
            ('water_level', 'cm', (0, 100), 'industrial control')
        ]
        
        while time.time() - start_time < duration_seconds:
            # Select sensor to attack
            sensor_type, unit, value_range, system = random.choice(sensor_types)
            
            # Generate normal and injected values
            normal_value = random.uniform(value_range[0] * 0.5, value_range[1] * 0.5)
            
            # Choose injection method
            injection_method = random.choice(['extreme_value', 'multiply', 'negate', 'freeze'])
            
            if injection_method == 'extreme_value':
                injected_value = random.choice([
                    value_range[0] - random.uniform(10, 50),  # Very low
                    value_range[1] + random.uniform(10, 50)   # Very high
                ])
                description = f"Injected extreme value outside normal range"
                
            elif injection_method == 'multiply':
                multiplier = random.uniform(2, 5)
                injected_value = normal_value * multiplier
                description = f"Multiplied value by {multiplier:.1f}x"
                
            elif injection_method == 'negate':
                if sensor_type in ['motion', 'door_status']:
                    injected_value = 1 - normal_value  # Flip binary value
                    description = "Flipped binary state"
                else:
                    injected_value = -normal_value
                    description = "Negated value"
                    
            else:  # freeze
                injected_value = normal_value  # Same value repeatedly
                description = "Frozen value (no changes)"
            
            attack_entry = {
                'timestamp': datetime.now(),
                'attack_type': 'data_injection',
                'sensor_type': sensor_type,
                'normal_value': f"{normal_value:.1f}{unit}",
                'injected_value': f"{injected_value:.1f}{unit}",
                'unit': unit,
                'target_system': system,
                'injection_method': injection_method,
                'description': description,
                'impact': random.choice(['False alarm', 'System malfunction', 'Wrong decision', 'Safety issue']),
                'is_attack': True,
                'subtlety': 'High' if injection_method == 'freeze' else 'Low'
            }
            
            attack_log.append(attack_entry)
            injection_count += 1
            
            # Print injection details
            elapsed = time.time() - start_time
            if verbose:
                print(f"  💉 Injection {injection_count}: {sensor_type.upper()} "
                      f"{attack_entry['normal_value']} → {attack_entry['injected_value']}")
                print(f"     Method: {injection_method} | Impact: {attack_entry['impact']}")
            
            time.sleep(random.uniform(3, 8))
        
        if verbose:
            print(f"\n✅ Data injection attack simulation complete!")
            print(f"   • Total injections: {injection_count}")
            print(f"   • Attack duration: {duration_seconds} seconds")
            print(f"   • Target systems: {len(set(a['target_system'] for a in attack_log))}")
            print(f"   • Detection: Statistical anomaly detection required")
            print(f"   • Prevention: Data validation, checksums, blockchain for integrity")
        
        return attack_log
    
    def run_comprehensive_simulation(self, total_duration=120, verbose=True):
        """Run comprehensive attack simulation with mixed attacks"""
        if verbose:
            print("\n" + "="*60)
            print("🎯 COMPREHENSIVE ATTACK SIMULATION LABORATORY")
            print("="*60)
            print(f"Total simulation time: {total_duration} seconds")
            print("Simulating normal traffic mixed with various attacks")
            print("="*60)
        
        all_traffic = []
        attack_schedule = [
            (0, 30, 'normal', "Phase 1: Normal operation"),
            (30, 60, 'dos', "Phase 2: DoS attack in progress"),
            (60, 90, 'normal', "Phase 3: Recovery from DoS"),
            (90, 105, 'mitm', "Phase 4: MITM attack intercepting"),
            (105, 120, 'injection', "Phase 5: Data injection attack")
        ]
        
        start_time = time.time()
        
        if verbose:
            print(f"\n⏰ Starting simulation at {datetime.now().strftime('%H:%M:%S')}")
            print(f"Schedule: 0-30s: Normal, 30-60s: DoS, 60-90s: Normal,")
            print(f"          90-105s: MITM, 105-120s: Data Injection")
        
        while time.time() - start_time < total_duration:
            elapsed = time.time() - start_time
            
            # Determine current phase
            current_phase = 'normal'
            phase_description = "Normal operation"
            
            for phase_start, phase_end, phase_type, description in attack_schedule:
                if phase_start <= elapsed < phase_end:
                    current_phase = phase_type
                    phase_description = description
                    break
            
            # Generate traffic based on current phase
            if verbose and elapsed < 5:  # Only show phase changes occasionally
                print(f"\n{'🚨' if current_phase != 'normal' else '📡'} {phase_description}")
            
            if current_phase == 'normal':
                # Generate 5 seconds of normal traffic
                traffic_batch = self.simulate_normal_traffic(
                    duration_seconds=5, verbose=False)
                all_traffic.extend(traffic_batch)
                
            elif current_phase == 'dos':
                # Generate DoS attack
                attack_batch = self.simulate_dos_attack(
                    duration_seconds=5, verbose=False)
                all_traffic.extend(attack_batch)
                
            elif current_phase == 'mitm':
                # Generate MITM attack
                attack_batch = self.simulate_mitm_attack(
                    duration_seconds=5, verbose=False)
                all_traffic.extend(attack_batch)
                
            elif current_phase == 'injection':
                # Generate data injection
                attack_batch = self.simulate_data_injection_attack(
                    duration_seconds=5, verbose=False)
                all_traffic.extend(attack_batch)
        
        # Analysis
        if verbose:
            print("\n" + "="*60)
            print("📊 SIMULATION COMPLETE - ANALYSIS")
            print("="*60)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_traffic)
        
        # Count attacks
        attack_count = 0
        normal_count = 0
        
        if 'is_attack' in df.columns:
            attack_count = df['is_attack'].sum()
            normal_count = len(df) - attack_count
        
        if verbose:
            print(f"\n📈 Traffic Statistics:")
            print(f"   • Total packets/events: {len(df):,}")
            print(f"   • Normal packets: {normal_count:,}")
            print(f"   • Attack packets: {attack_count:,}")
            print(f"   • Attack percentage: {attack_count/len(df)*100:.1f}%")
        
        # Analyze attack types
        attack_types = {}
        if 'attack_type' in df.columns:
            attack_series = df[df['is_attack'] == True]['attack_type']
            attack_types = attack_series.value_counts().to_dict()
        
        if verbose and attack_types:
            print(f"\n🎯 Attack Type Distribution:")
            for attack_type, count in attack_types.items():
                percentage = count / attack_count * 100 if attack_count > 0 else 0
                print(f"   • {attack_type}: {count:,} ({percentage:.1f}%)")
        
        # Timing analysis
        if 'timestamp' in df.columns and len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
            
            if verbose:
                print(f"\n⏱️  Timing Analysis:")
                print(f"   • Simulation duration: {duration:.1f} seconds")
                print(f"   • Average event rate: {len(df)/duration:.1f} events/second")
                
                if attack_count > 0:
                    attack_df = df[df['is_attack'] == True]
                    attack_duration = (attack_df['timestamp'].max() - attack_df['timestamp'].min()).total_seconds()
                    print(f"   • Attack duration: {attack_duration:.1f} seconds")
                    print(f"   • Attack intensity: {attack_count/attack_duration:.1f} attacks/second")
        
        if verbose:
            print("\n" + "="*60)
            print("✅ Comprehensive simulation complete!")
            print("="*60)
            print("\n🎓 Educational Insights:")
            print("1. Different attacks have different patterns and indicators")
            print("2. DoS attacks are high-volume but easy to detect")
            print("3. MITM attacks are stealthy but leave protocol anomalies")
            print("4. Data injection requires understanding of normal data patterns")
            print("5. A good IDS must detect all these attack types")
        
        return {
            'total_events': len(df),
            'normal_count': normal_count,
            'attack_count': attack_count,
            'attack_types': attack_types,
            'dataframe': df
        }

def print_menu():
    """Print simulation menu"""
    print("\n" + "="*60)
    print("🔬 ADVANCED IOT ATTACK SIMULATION LABORATORY")
    print("="*60)
    print("\nSimulate various IoT attacks to test your IDS")
    print("\nSelect simulation type:")
    print("  1. Normal IoT traffic (30s)")
    print("  2. Denial of Service attack (20s)")
    print("  3. Man-in-the-Middle attack (25s)")
    print("  4. Data injection attack (20s)")
    print("  5. Comprehensive simulation (120s - all attacks)")
    print("  6. View attack profiles")
    print("  7. Exit")
    print("="*60)

def main():
    """Main function to run attack simulations"""
    simulator = AdvancedAttackSimulator()
    
    while True:
        print_menu()
        
        try:
            choice = input("\nEnter choice (1-7): ").strip()
            
            if choice == '1':
                simulator.simulate_normal_traffic(30)
                
            elif choice == '2':
                simulator.simulate_dos_attack(20)
                
            elif choice == '3':
                simulator.simulate_mitm_attack(25)
                
            elif choice == '4':
                simulator.simulate_data_injection_attack(20)
                
            elif choice == '5':
                results = simulator.run_comprehensive_simulation(120)
                
                # Save results to file
                import os
                os.makedirs("outputs/simulations", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = f"outputs/simulations/simulation_{timestamp}.txt"
                
                with open(report_path, 'w') as f:
                    f.write("IoT Attack Simulation Report\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Simulation Time: {datetime.now()}\n")
                    f.write(f"Total Events: {results['total_events']}\n")
                    f.write(f"Normal Events: {results['normal_count']}\n")
                    f.write(f"Attack Events: {results['attack_count']}\n\n")
                    
                    f.write("Attack Type Distribution:\n")
                    for attack_type, count in results['attack_types'].items():
                        f.write(f"  {attack_type}: {count}\n")
                
                print(f"\n📄 Detailed report saved to: {report_path}")
                
            elif choice == '6':
                print("\n📚 ATTACK PROFILES:")
                print("="*60)
                for attack_id, profile in simulator.attack_profiles.items():
                    print(f"\n{profile['name'].upper()}:")
                    print(f"  Description: {profile['description']}")
                    print(f"  Difficulty: {profile['difficulty']}")
                    print(f"  Impact: {profile['impact']}")
                    print(f"  Target: {profile['target']}")
                    print(f"  Detection: {profile['detection']}")
                    print(f"  Indicators: {', '.join(profile['indicators'][:3])}")
                
            elif choice == '7':
                print("\n👋 Exiting simulation laboratory...")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-7.")
            
            # Pause before showing menu again
            if choice != '7':
                input("\nPress Enter to continue...")
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Simulation stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    print("\n" + "="*60)
    print("🎯 Simulation laboratory closed")
    print("="*60)


if __name__ == "__main__":
    main()