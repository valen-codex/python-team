# Detect suspicious IP addresses from log file
import pandas as pd
df = pd.read_csv('log_data.csv')
suspicious_ips = df[df['threat_level'] == 'High']['source_ip'].value_counts()
# suspicious_ips = suspicious_ips[suspicious_ips > 100]  
# print(suspicious_ips.head(10))
print(suspicious_ips)
