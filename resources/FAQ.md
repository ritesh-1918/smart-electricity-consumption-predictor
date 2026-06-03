# Volterra: Frequently Asked Questions (FAQ)

Here are the answers to the most common questions raised during the live Q&A session:

---

### Q1: Why did we map `Weekday` to `0` and `Weekend` to `1`?
**A**: Machine learning models calculate relationships using mathematical equations. They cannot multiply or add string categories like `'Weekday'`. We map categories to numbers (`0` and `1`) to make them compatible with arithmetic linear equations.

### Q2: What is the difference between Feature Weights (Coefficients) and Feature Contributions?
*   **Feature Weights (Coefficients)**: The fixed slope learned during training. It represents the impact of a 1-unit increase in that feature (e.g. AC Hours coefficient is `+4.45`). This value is static.
*   **Feature Contributions**: The weight multiplied by the *current* slider value (e.g. $4.45 \times 10\text{ hours} = 44.5\text{ kWh}$). This value changes in real-time as you drag the inputs.

### Q3: What does the R-squared ($R^2$) score of 94.95% mean?
**A**: The $R^2$ score (Coefficient of Determination) measures how well the model's predictions match actual observations. An $R^2$ of 94.95% means the 6 features we inputted explain 94.95% of the variance in household electricity consumption. The remaining 5.05% is unmodeled random noise.

### Q4: Why is the model saved as a `.pkl` file?
**A**: Pickle (`.pkl`) is a standard Python serialization format. It allows us to save the trained model object (the mathematical equation parameters) directly to disk and reload it instantly in the Streamlit application without retraining the model.

### Q5: Why did we deploy the app using Docker instead of a standard GitHub Space?
**A**: Using a Docker container guarantees that the server runs on the exact same Python version (`3.10`) and system configuration in production as it does locally, preventing "works on my machine" deployment crashes.
