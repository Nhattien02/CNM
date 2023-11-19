from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Load mô hình và dữ liệu
model = LinearRegression()
data = pd.read_csv("housing.csv")
data = data.drop(columns="ocean_proximity")
data = data.dropna()
X = data[["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population",
          "households", "median_income"]]
y = data["median_house_value"]
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ yêu cầu POST
    input_data = request.json

    # Chuyển đổi dữ liệu thành DataFrame để dễ xử lý
    input_df = pd.DataFrame([input_data])

    # Dự đoán giá nhà
    predicted_price = model.predict(input_df)

    # Trả về kết quả dưới dạng JSON
    return jsonify({"predicted_price": predicted_price[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
    #app.run(debug=True)
    #app.run(port=8000)
