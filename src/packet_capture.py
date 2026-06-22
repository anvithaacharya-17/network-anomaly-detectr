"""
This module captures network packets using Scapy.
Think of it like a security camera recording network traffic.
"""

from scapy.all import sniff, IP, TCP, UDP, ICMP
import time
import csv
import os
from datetime import datetime

class NetworkCapture:
    """
    Captures network packets and saves them.
    """
    
    def __init__(self, interface=None):
        """
        Initialize the capture.
        
        Parameters:
        - interface: Which network card to use (None = auto-detect)
        """
        self.interface = interface
        self.packets = []
        self.captured_count = 0
        
    def capture_packets(self, count=100, timeout=30):
        """
        Capture a specified number of packets.
        
        Parameters:
        - count: Number of packets to capture
        - timeout: Maximum time to capture (seconds)
        
        Returns:
        - List of captured packets
        """
        print(f"Starting packet capture... Capture {count} packets or wait {timeout}s")
        print("Press Ctrl+C to stop early")
        
        try:
            # This sniffs network traffic
            packets = sniff(
                count=count,          # Stop after this many packets
                timeout=timeout,      # Or stop after this many seconds
                iface=self.interface, # Which network interface to use
                prn=self.process_packet  # Process each packet as it's captured
            )
            
            print(f"\nCapture complete! Captured {len(packets)} packets")
            return packets
            
        except KeyboardInterrupt:
            print("\nCapture stopped by user")
            return self.packets
        except Exception as e:
            print(f"Error during capture: {e}")
            return []
    
    def process_packet(self, packet):
        """
        Process each packet as it's captured.
        Extracts relevant information.
        """
        self.captured_count += 1
        
        # Show progress every 10 packets
        if self.captured_count % 10 == 0:
            print(f"Captured {self.captured_count} packets...")
        
        # Only process IP packets
        if packet.haslayer(IP):
            ip = packet[IP]
            
            # Extract basic info
            packet_info = {
                'timestamp': time.time(),
                'src_ip': ip.src,
                'dst_ip': ip.dst,
                'protocol': ip.proto,
                'ttl': ip.ttl,
                'length': len(packet),  # Packet size in bytes
                'flags': '',
                'src_port': None,
                'dst_port': None,
                'tcp_flags': ''
            }
            
            # Check if it's TCP
            if packet.haslayer(TCP):
                tcp = packet[TCP]
                packet_info['src_port'] = tcp.sport
                packet_info['dst_port'] = tcp.dport
                packet_info['flags'] = 'TCP'
                packet_info['tcp_flags'] = str(tcp.flags)
                
            # Check if it's UDP
            elif packet.haslayer(UDP):
                udp = packet[UDP]
                packet_info['src_port'] = udp.sport
                packet_info['dst_port'] = udp.dport
                packet_info['flags'] = 'UDP'
                
            # Check if it's ICMP
            elif packet.haslayer(ICMP):
                packet_info['flags'] = 'ICMP'
                packet_info['src_port'] = None
                packet_info['dst_port'] = None
            
            self.packets.append(packet_info)
            
    def save_to_csv(self, filename=None):
        """
        Save captured packets to a CSV file.
        
        Parameters:
        - filename: Name of the file (auto-generate if None)
        """
        if not self.packets:
            print("No packets to save!")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/capture_{timestamp}.csv"
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Write to CSV
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'src_ip', 'dst_ip', 'protocol', 
                         'src_port', 'dst_port', 'ttl', 'length', 'flags', 'tcp_flags']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.packets)
        
        print(f"Saved {len(self.packets)} packets to {filename}")
        return filename
    
    def clear(self):
        """Clear captured packets from memory"""
        self.packets = []
        self.captured_count = 0


def quick_capture(output_file=None, count=100):
    """
    Quick function to capture and save packets.
    
    Parameters:
    - output_file: CSV file to save to
    - count: Number of packets to capture
    """
    capture = NetworkCapture()
    packets = capture.capture_packets(count=count, timeout=30)
    
    if packets:
        filename = capture.save_to_csv(output_file)
        return filename
    else:
        print("No packets captured!")
        return None

if __name__ == "__main__":
    # Test the capture
    print("Testing packet capture...")
    print("This will capture 50 packets or stop after 20 seconds")
    filename = quick_capture(count=50)
    
    if filename:
        print(f"Success! Check {filename} for the captured packets")
    else:
        print("No packets captured. Make sure you're connected to the internet!")