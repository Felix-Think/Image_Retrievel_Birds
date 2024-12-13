
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
#|%%--%%| <wifEmOrQah|xHhRuJe0vY>

# Load data
data = pd.read_csv('DATA_BIRDS.csv')
data_1 = data.drop(columns=['Tên loài', 'Loại rừng', 'Mùa'])
processing = data_1.copy()
#|%%--%%| <xHhRuJe0vY|8MYQPXntoD>

# Label encoding for categorical features
label = LabelEncoder()
categorical_features = processing.select_dtypes(include=['object']).columns
for col in categorical_features:
    processing[col] = label.fit_transform(processing[col])
#|%%--%%| <8MYQPXntoD|YVjV5aUQVr>

# Adjust 'Tỷ lệ sinh sản' based on other columns
processing['Tỷ lệ sinh sản'] += data['Di cư'].apply(lambda x: 2 if x == 0 else 0)
processing['Tỷ lệ sinh sản'] -= data['Rủi ro săn mồi'].apply(lambda x: 1 if x == 0 else 0.5 if x == 2 else 0.2)
#|%%--%%| <YVjV5aUQVr|naGWXtHPhP>

# Calculate composite score
processing['Chỉ số tổ hợp'] = (
    0.35 * processing['Số lượng'] + 
    0.15 * processing['Diện tích sinh sống (km²)'] + 
    0.25 * processing['Tỷ lệ sinh sản'] + 
    0.15 * processing['Mối đe dọa môi trường'] + 
    0.10 * (100 - processing['Đô thị hóa'])
)
#|%%--%%| <naGWXtHPhP|alzPCePfVF>

# Normalize the composite score
scaler = StandardScaler()
processing['Chỉ số tổ hợp'] = scaler.fit_transform(processing[['Chỉ số tổ hợp']])
#|%%--%%| <alzPCePfVF|l42iBAPUw7>

# Label assignment function
def assign_label(score):
    if score <= -1:
        if -0.67777 <= score <= -0.54345:
            return 'Nguy cơ trung bình'
        elif -0.72323 <= score < -0.67778:
            return 'Nguy cơ thấp'
        elif -0.766 <= score < -0.72324:
            return 'Nguy cơ cao'
        else:
            return random.choice(['Không có nguy cơ', 'Nguy cơ trung bình'])
    else:
        if 0.2 <= score < 0.5:
            return np.random.choice(['Nguy cơ trung bình', 'Nguy cơ thấp'])
        elif 0.5 <= score < 0.7:
            return 'Nguy cơ trung bình'
        elif 0.7 <= score <= 1.1:
            return 'Nguy cơ cao'
        elif score >= 1.1:
            return random.choice(['Nguy cơ thấp', 'Nguy cơ trung bình'])
        else:
            return random.choice(['Không có nguy cơ', 'Nguy cơ thấp', 'Nguy cơ cao', 'Nguy cơ trung bình'])
#|%%--%%| <l42iBAPUw7|jFal521Luk>

# Apply the label assignment function to the composite score
processing['Nguy cơ tuyệt chủng'] = processing['Chỉ số tổ hợp'].apply(assign_label)
#|%%--%%| <jFal521Luk|9fkgIQodCV>

# Print summary of the labels
print(processing['Nguy cơ tuyệt chủng'].value_counts())
#|%%--%%| <9fkgIQodCV|qzeD7CI4CB>

# Shuffle data if needed and save to file
data['Nguy cơ tuyệt chủng'] = processing['Nguy cơ tuyệt chủng']
data = data.sample(frac=1).reset_index(drop=True)
data.to_csv('DATA_BIRDS_PROCESSED.csv', index=False)

