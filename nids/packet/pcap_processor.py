import os
import logging
from nfstream import NFStreamer
import pandas as pd
import requests
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")

API_URL = "http://localhost:8000/api/v1/predict/batch"

class PCAPProcessor:
    def __init__(self, pcap_path):
        self.pcap_path = pcap_path
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"PCAP not found: {pcap_path}")
            
    def process(self):
        logging.info(f"Extracting flows from {self.pcap_path} using NFStream...")
        start_time = time.time()
        
        # NFStream extracts many features out of the box that map closely to CIC-IDS2017
        streamer = NFStreamer(source=self.pcap_path, statistical_analysis=True)
        flows = []
        
        for flow in streamer:
            # Map NFStream features to CIC-IDS2017 feature names
            # (Note: NFStream doesn't extract all 78 perfectly identically, but many are very close.
            # A robust pipeline would require a precise mapping layer. Here we map essential metrics.)
            f_dict = {
                "source_ip": flow.src_ip,
                "destination_ip": flow.dst_ip,
                "Destination_Port": flow.dst_port,
                "protocol_str": str(flow.protocol),
                "Flow_Duration": flow.bidirectional_duration_ms * 1000, # ms to microsec
                "Total_Fwd_Packets": flow.src2dst_packets,
                "Total_Backward_Packets": flow.dst2src_packets,
                "Total Length of Fwd Packets": flow.src2dst_bytes,
                "Total Length of Bwd Packets": flow.dst2src_bytes,
                "Fwd Packet Length Max": flow.src2dst_max_ps,
                "Fwd Packet Length Min": flow.src2dst_min_ps,
                "Fwd Packet Length Mean": flow.src2dst_mean_ps,
                "Fwd Packet Length Std": flow.src2dst_stddev_ps,
                "Bwd Packet Length Max": flow.dst2src_max_ps,
                "Bwd Packet Length Min": flow.dst2src_min_ps,
                "Bwd Packet Length Mean": flow.dst2src_mean_ps,
                "Bwd Packet Length Std": flow.dst2src_stddev_ps,
            }
            flows.append(f_dict)
            
        logging.info(f"Extracted {len(flows)} flows in {time.time() - start_time:.2f}s")
        return flows
        
    def send_to_api(self, flows, batch_size=500):
        logging.info(f"Sending {len(flows)} flows to API...")
        for i in range(0, len(flows), batch_size):
            batch = flows[i:i+batch_size]
            try:
                res = requests.post(API_URL, json=batch)
                res.raise_for_status()
                logging.info(f"Successfully sent batch {i//batch_size + 1}")
            except Exception as e:
                logging.error(f"Failed to send batch {i//batch_size + 1}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PCAP to NIDS API")
    parser.add_argument("pcap_file", help="Path to PCAP file")
    args = parser.parse_args()
    
    processor = PCAPProcessor(args.pcap_file)
    flows = processor.process()
    processor.send_to_api(flows)
