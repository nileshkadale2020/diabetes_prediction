# Example Input Values for Diabetes Prediction Application

This document provides example input values for demonstrating the diabetes prediction web application during your presentation.

---

## Understanding the Input Fields

### Medical Parameters Explained:

1. **Pregnancies** (0-20): Number of times pregnant
2. **Glucose** (0-300 mg/dL): Plasma glucose concentration after 2 hours in an oral glucose tolerance test
3. **Blood Pressure** (0-150 mm Hg): Diastolic blood pressure
4. **Skin Thickness** (0-100 mm): Triceps skin fold thickness
5. **Insulin** (0-1000 mu U/ml): 2-Hour serum insulin
6. **BMI** (0-60): Body Mass Index (weight in kg / (height in m)²)
7. **Diabetes Pedigree Function** (0-3): A function that scores likelihood of diabetes based on family history
8. **Age** (0-120 years): Age in years

---

## Example Scenarios

### Example 1: High Risk Patient
**Scenario**: Middle-aged woman with elevated glucose and BMI

```
Pregnancies: 6
Glucose: 148
Blood Pressure: 72
Skin Thickness: 35
Insulin: 0
BMI: 33.6
Diabetes Pedigree Function: 0.627
Age: 50
```

**Expected Result**: 
- Risk Level: **High**
- Probability: **~70-85%**
- Interpretation: "Based on the input parameters, there is a risk of diabetes. Please consult with a healthcare professional."

**Why High Risk:**
- Elevated glucose (148 is above normal)
- High BMI (33.6 indicates obesity)
- Multiple pregnancies (6)
- Family history (pedigree function 0.627)
- Age 50 (higher risk age group)

---

### Example 2: Low Risk Patient
**Scenario**: Young, healthy individual with normal values

```
Pregnancies: 1
Glucose: 85
Blood Pressure: 66
Skin Thickness: 29
Insulin: 0
BMI: 26.6
Diabetes Pedigree Function: 0.351
Age: 31
```

**Expected Result**:
- Risk Level: **Low**
- Probability: **~15-30%**
- Interpretation: "Based on the input parameters, the risk of diabetes appears to be low. However, regular check-ups are recommended."

**Why Low Risk:**
- Normal glucose (85 is within normal range)
- Healthy BMI (26.6 is slightly overweight but acceptable)
- Young age (31)
- Lower family history risk (0.351)
- Normal blood pressure (66)

---

### Example 3: Borderline Case (Low Risk)
**Scenario**: Middle-aged person with slightly elevated values

```
Pregnancies: 3
Glucose: 120
Blood Pressure: 70
Skin Thickness: 30
Insulin: 100
BMI: 28.5
Diabetes Pedigree Function: 0.450
Age: 45
```

**Expected Result**:
- Risk Level: **Low** (probability typically 35-45%)
- Probability: **~35-45%**
- Interpretation: "Based on the input parameters, the risk of diabetes appears to be low. However, regular check-ups are recommended."
- **Note**: The app uses a two-tier system (High ≥50%, Low <50%). This borderline case typically falls in the Low category.

**Why Borderline:**
- Slightly elevated glucose (120 is pre-diabetic range)
- Overweight BMI (28.5)
- Moderate family history (0.450)
- Age 45 (approaching higher risk age)
- Despite these factors, the model typically predicts below 50% probability, classifying it as Low risk

---

### Example 4: Very High Risk Patient
**Scenario**: Older person with multiple risk factors

```
Pregnancies: 8
Glucose: 180
Blood Pressure: 90
Skin Thickness: 40
Insulin: 0
BMI: 35.0
Diabetes Pedigree Function: 0.800
Age: 60
```

**Expected Result**:
- Risk Level: **High**
- Probability: **~85-95%**
- Interpretation: "Based on the input parameters, there is a significant risk of diabetes. Please consult with a healthcare professional immediately."

**Why Very High Risk:**
- Very high glucose (180 is diabetic range)
- Obese BMI (35.0)
- High blood pressure (90)
- Strong family history (0.800)
- Older age (60)
- Multiple pregnancies (8)

---

### Example 5: Young, Healthy Individual
**Scenario**: Very low risk case

```
Pregnancies: 0
Glucose: 75
Blood Pressure: 60
Skin Thickness: 20
Insulin: 50
BMI: 22.0
Diabetes Pedigree Function: 0.200
Age: 25
```

**Expected Result**:
- Risk Level: **Low**
- Probability: **~5-15%**
- Interpretation: "Based on the input parameters, the risk of diabetes appears to be very low."

**Why Very Low Risk:**
- Normal glucose (75)
- Healthy BMI (22.0)
- Young age (25)
- Low family history (0.200)
- Normal blood pressure (60)
- No pregnancies

---

## Presentation Tips

### During Demo:

1. **Start with Example 2 (Low Risk)**: 
   - Shows the system working correctly
   - Demonstrates low-risk assessment
   - Builds confidence in the system

2. **Then show Example 1 (High Risk)**:
   - Demonstrates the system identifying high-risk patients
   - Shows the difference in probability scores
   - Highlights the clinical relevance

3. **If time permits, show Example 3 (Borderline)**:
   - Demonstrates nuanced predictions
   - Shows how the system handles uncertain cases
   - Explains why multiple models are useful

### Key Points to Emphasize:

1. **Realistic Values**: All examples use realistic medical values
2. **Multiple Risk Factors**: Examples show how combinations of factors affect risk
3. **Probability Scores**: Explain what the percentage means
4. **Medical Disclaimer**: Always remind that this is for educational purposes

### Explaining the Results:

**When showing results, explain:**
- "The probability of [X]% means that based on the training data, [X]% of patients with similar characteristics were diagnosed with diabetes"
- "The risk level (High/Low) is determined by a threshold of 50% probability"
- "We show predictions from all models, but use the best-performing model for the final assessment"
- "The system considers all factors together, not just individual values"

---

## Notes for Presentation

### Important Disclaimers to Mention:

1. **Educational Purpose**: "This application is for educational and demonstration purposes only"
2. **Not Medical Advice**: "It should not be used as a substitute for professional medical advice, diagnosis, or treatment"
3. **Consult Healthcare Professionals**: "Always consult with qualified healthcare professionals for medical decisions"
4. **Dataset Limitations**: "The model was trained on a specific dataset and may not generalize to all populations"

### Technical Details to Highlight:

1. **Preprocessing**: "The same preprocessing pipeline used during training is applied to user inputs"
2. **Model Ensemble**: "We use multiple models and select the best one for final prediction"
3. **Real-time Processing**: "Predictions are generated instantly using pre-loaded models"
4. **Consistency**: "The preprocessing ensures consistency between training and prediction"

---

## Additional Test Cases

### Edge Cases (for testing, not presentation):

**Very Low Values:**
```
Pregnancies: 0
Glucose: 50
Blood Pressure: 50
Skin Thickness: 10
Insulin: 0
BMI: 18.0
Diabetes Pedigree Function: 0.100
Age: 20
```

**Very High Values:**
```
Pregnancies: 15
Glucose: 250
Blood Pressure: 120
Skin Thickness: 60
Insulin: 500
BMI: 45.0
Diabetes Pedigree Function: 1.500
Age: 80
```

**Mixed Signals:**
```
Pregnancies: 2
Glucose: 140 (high)
Blood Pressure: 65 (normal)
Skin Thickness: 25 (normal)
Insulin: 0
BMI: 24.0 (normal)
Diabetes Pedigree Function: 0.300 (low)
Age: 35 (young)
```

---

**Use these examples to create an engaging and informative demonstration!**

