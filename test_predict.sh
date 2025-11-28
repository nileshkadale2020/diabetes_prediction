#!/bin/bash

# Test script for the Diabetes Prediction Flask API
# Tests multiple scenarios to validate the /predict endpoint

BASE_URL="http://127.0.0.1:5000"
PREDICT_ENDPOINT="${BASE_URL}/predict"

echo "=========================================="
echo "Diabetes Prediction API Test Suite"
echo "=========================================="
echo ""

# Test 1: High-risk case (original test case)
echo "Test 1: High-risk patient"
echo "Input: 6 pregnancies, 148 glucose, 72 BP, 35 skin thickness, 0 insulin, 33.6 BMI, 0.627 DPF, 50 age"
curl -s -X POST "$PREDICT_ENDPOINT" \
  -d 'pregnancies=6' \
  -d 'glucose=148' \
  -d 'blood_pressure=72' \
  -d 'skin_thickness=35' \
  -d 'insulin=0' \
  -d 'bmi=33.6' \
  -d 'diabetes_pedigree=0.627' \
  -d 'age=50' | python -m json.tool
echo ""
echo "---"
echo ""

# Test 2: Low-risk case
echo "Test 2: Low-risk patient"
echo "Input: 1 pregnancy, 85 glucose, 66 BP, 29 skin thickness, 0 insulin, 26.6 BMI, 0.35 DPF, 31 age"
curl -s -X POST "$PREDICT_ENDPOINT" \
  -d 'pregnancies=1' \
  -d 'glucose=85' \
  -d 'blood_pressure=66' \
  -d 'skin_thickness=29' \
  -d 'insulin=0' \
  -d 'bmi=26.6' \
  -d 'diabetes_pedigree=0.35' \
  -d 'age=31' | python -m json.tool
echo ""
echo "---"
echo ""

# Test 3: Borderline case
echo "Test 3: Borderline patient (near decision threshold)"
echo "Input: 2 pregnancies, 100 glucose, 70 BP, 32 skin thickness, 50 insulin, 28.0 BMI, 0.45 DPF, 45 age"
curl -s -X POST "$PREDICT_ENDPOINT" \
  -d 'pregnancies=2' \
  -d 'glucose=100' \
  -d 'blood_pressure=70' \
  -d 'skin_thickness=32' \
  -d 'insulin=50' \
  -d 'bmi=28.0' \
  -d 'diabetes_pedigree=0.45' \
  -d 'age=45' | python -m json.tool
echo ""
echo "---"
echo ""

# Test 4: Extreme values (very high)
echo "Test 4: Extreme high-risk case"
echo "Input: 10 pregnancies, 200 glucose, 100 BP, 50 skin thickness, 300 insulin, 45.0 BMI, 1.0 DPF, 65 age"
curl -s -X POST "$PREDICT_ENDPOINT" \
  -d 'pregnancies=10' \
  -d 'glucose=200' \
  -d 'blood_pressure=100' \
  -d 'skin_thickness=50' \
  -d 'insulin=300' \
  -d 'bmi=45.0' \
  -d 'diabetes_pedigree=1.0' \
  -d 'age=65' | python -m json.tool
echo ""
echo "---"
echo ""

# Test 5: Young adult, minimal risk factors
echo "Test 5: Young adult with minimal risk factors"
echo "Input: 0 pregnancies, 70 glucose, 60 BP, 20 skin thickness, 0 insulin, 22.0 BMI, 0.1 DPF, 21 age"
curl -s -X POST "$PREDICT_ENDPOINT" \
  -d 'pregnancies=0' \
  -d 'glucose=70' \
  -d 'blood_pressure=60' \
  -d 'skin_thickness=20' \
  -d 'insulin=0' \
  -d 'bmi=22.0' \
  -d 'diabetes_pedigree=0.1' \
  -d 'age=21' | python -m json.tool
echo ""
echo "---"
echo ""

# Test 6: Older patient with high risk factors
echo "Test 6: Older patient with high risk factors"
echo "Input: 5 pregnancies, 150 glucose, 85 BP, 40 skin thickness, 100 insulin, 38.0 BMI, 0.8 DPF, 72 age"
curl -s -X POST "$PREDICT_ENDPOINT" \
  -d 'pregnancies=5' \
  -d 'glucose=150' \
  -d 'blood_pressure=85' \
  -d 'skin_thickness=40' \
  -d 'insulin=100' \
  -d 'bmi=38.0' \
  -d 'diabetes_pedigree=0.8' \
  -d 'age=72' | python -m json.tool
echo ""
echo "---"
echo ""

# Test 7: Check homepage (should return HTML)
echo "Test 7: Check homepage accessibility"
echo "GET /"
HOMEPAGE_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/")
echo "HTTP Status: $HOMEPAGE_STATUS"
if [ "$HOMEPAGE_STATUS" = "200" ]; then
  echo "✓ Homepage is accessible"
else
  echo "✗ Homepage failed"
fi
echo ""
echo "---"
echo ""

# Test 8: Check diabetes prediction page
echo "Test 8: Check diabetes prediction page"
echo "GET /diabetes"
DIABETES_PAGE_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/diabetes")
echo "HTTP Status: $DIABETES_PAGE_STATUS"
if [ "$DIABETES_PAGE_STATUS" = "200" ]; then
  echo "✓ Diabetes page is accessible"
else
  echo "✗ Diabetes page failed"
fi
echo ""

echo "=========================================="
echo "Test Suite Complete"
echo "=========================================="
