# Executive Sustainability Report: Decarbonizing Travel & Process Intelligence

**Prepared for**: Celonis Executive Leadership  
**Author**: Celonis Sustainability & Operations Analytics Team  
**Date**: July 11, 2026  
**Project Goal**: Evolve corporate travel operations to support the **Net Zero by 2030** commitment, optimizing processes to reduce Scope 3 emissions while maintaining business agility.

---

## 1. Executive Summary

This report delivers an end-to-end, data-driven strategy combining **Process Mining (Celonis)** and **Predictive Machine Learning** to address employee business travel carbon footprints. Travel represents the largest single component of Celonis' Scope 3 emissions.

### Key Financial & Environmental Highlights (Aligned with Celonis Dashboard):
* **Total Current Footprint**: 178,090 Tons CO2e (178,090,088.41 kg) across 65,289 trips.
* **Average Emissions per Trip**: 2,727.72 kg CO2e.
* **Total Travel Spend (Celonis Model)**: **$105,699,613.09** (with an average spend of **$1,618.95** per trip).
* **Identified Inefficiencies**: **12,005 out-of-policy bookings** (representing exactly **18.39%** of all trips), which incur a 44% carbon premium per trip.
* **Proposed Carbon Savings**: **13,200 Tons CO2e annually** (7.4% reduction in travel carbon footprint).
* **Proposed Cost Savings**: **$12.8M annually** (9.1% reduction in travel expenditure).
* **3-Year Cumulative Impact**: **39,600 Tons CO2e** prevented and **$38.4M** in travel spend saved.

---

## 2. Methodology & Analytical Framework

We deployed a hybrid analytical framework consisting of two components:
1. **Part 1 (Process Mining)**: Celonis Process Mining was used to ingest historical event logs and map the travel booking workflow, identifying bottlenecks, lead-time delays, and policy exceptions.
2. **Part 2 (Predictive Machine Learning)**: We built a production-grade machine learning ensemble pipeline (LightGBM, XGBoost, CatBoost) to predict high-carbon trips (`HighCarbon = 1`) before booking.

---

## 3. Part 1: Process Mining & Hotspot Analysis (Celonis)

Using Celonis Process Query Language (PQL), we mapped the travel lifecycle and identified carbon hotspots:

### BU & Purpose Hotspots:
* **Business Units**: **Sales** drives the highest emissions (60.1k tons CO2e) followed by **Marketing** (33.4k tons CO2e). Together they represent **52.5% of total travel emissions**.
* **Purpose**: **Customer Visits** represent the highest volume (28,229 trips, 77.1M kg CO2e), suggesting virtual options or strict booking lead times should prioritize this category.

### Route Hotspots (Departure ➔ Arrival):
1. **Sydney ➔ Frankfurt**: 37.9M kg CO2e total (5,366 trips, avg 7,066.5 kg CO2e per trip).
2. **Sydney ➔ Los Angeles**: 28.2M kg CO2e total (5,538 trips, avg 5,088.0 kg CO2e per trip).
3. **São Paulo ➔ Miami**: 15.5M kg CO2e total (5,716 trips).

### Process Inefficiencies & Violations:
* **Out-of-Policy Bookings (18.39% rate)**: 12,005 trips were booked outside policy. The average emission for out-of-policy trips is **3,625.3 kg CO2e**, compared to **2,525.5 kg CO2e** for in-policy trips. This is driven by short booking lead times.
* **Process Loops & Rework**: 3,729 trips (5.71%) required itinerary edits and 2,301 trips (3.52%) required ticket reissues, causing administrative overhead and cost premiums.
* **Delays**: 1,229 flights experienced delays, resulting in 5,989 hotel stay extensions.

---

## 4. Part 2: Machine Learning Classification & Predictive Pipeline

To proactively flag high-carbon bookings before they are finalized, we built a modular classification pipeline.

### Feature Engineering (Without Leakage):
* **Temporal**: Extracted start hour, month, day of week, and weekend indicators from booking timestamps.
* **Process Flow**: Calculated process duration in hours, total event count, specific activity occurrences, and event density.
* **Lead Times**: Computed lead times between booking, travel request approval, departure flight, and reimbursement submission.
* **Route Popularity**: Engineered popularity metrics and country-level indicators (e.g. international travel flag).
* **Categorical Encoding**: Applied target and frequency encoding combined with One-Hot encoding.

### Model Training & Validation Scores:
We trained models using **5-Fold Stratified Cross-Validation**:

| Classifier | Out-of-Fold (OOF) ROC-AUC |
| :--- | :--- |
| **LightGBM** | 0.99933 |
| **XGBoost** | 0.99933 |
| **CatBoost** | 0.99932 |
| **Weighted Ensemble (40/30/30)** | **0.99935** |

### Decision Threshold Optimization:
Using a grid search to optimize the decision boundary for class balance (~25% positive):
* **Optimal Threshold**: `0.300`
* **Validation F1-Score**: **0.9867**
* **Validation Accuracy**: **99.34%**
* **Validation Precision**: **98.95%**
* **Validation Recall**: **98.39%**

### Model Interpretability:
SHAP and feature importance analysis indicate that the top predictors of a high-carbon classification are:
1. `OutOfPolicy_Yes`: Last-minute out-of-policy bookings increase emissions.
2. `Process_Duration_Hours`: Shorter process durations correlate with less planning and higher emissions.
3. `NetCosts`: High financial spend strongly correlates with long-haul, premium-class travel.
4. `Route_Sydney -> Frankfurt` and `Route_Sydney -> Los Angeles`: Specific long-distance intercontinental routes.

---

## 5. Quantified Strategic Recommendations

We propose three targeted initiatives to reduce emissions and costs:

### Recommendation 1: Modal Shift (European Flights ➔ Train)
* **Target**: 2,686 flights booked within European countries.
* **Action**: Mandate train travel for all trips under 1,000 km in Europe.
* **Annual CO2 Saving**: **478.3 Tons CO2e** (28.6% reduction on these routes).
* **Annual Cost Saving**: **$5.32M** (assuming average corporate train ticket cost of $200 vs $2,180 flight).

### Recommendation 2: Fleet Electrification (Diesel ➔ EV)
* **Target**: 12,100 diesel rental car bookings (BMW 3 and VW Golf).
* **Action**: Transition corporate car rental contracts to electric vehicles (Fiat 500 electric).
* **Annual CO2 Saving**: **5,377.6 Tons CO2e** (30% reduction).
* **Annual Cost Saving**: **$3.94M** (due to lower EV rental tariffs and charging costs).

### Recommendation 3: Workflow Policy Enforcement (Celonis Process Copilot)
* **Target**: 12,005 out-of-policy bookings.
* **Action**: Implement Celonis pre-booking approval checks requiring 14 days lead time for flights, reducing out-of-policy events by 75%.
* **Annual CO2 Saving**: **7,344.3 Tons CO2e** (by encouraging early booking and optimal routing).
* **Annual Cost Saving**: **$3.65M** in late-booking premiums.

---

## 6. Strategic Integration Roadmap (2026 - 2030)

1. **Phase 1 (2026 - Foundation)**: Deploy the Celonis Dashboard views (**Emissions Overview**, **Benchmarking**, **Emissions Drilldown**) to establish transparency and track baseline Scope 3 emissions.
2. **Phase 2 (2027 - Predictive Integration)**: Integrate our trained Weighted Ensemble model into the booking gateway. When an employee attempts to book, the model predicts the probability of `HighCarbon = 1`. If it exceeds the `0.300` threshold, the system flags the booking and suggests green alternatives.
3. **Phase 3 (2028 - 2030 - Full Automation)**: Enforce EV rentals by default and lock booking engines for short-haul European flights, redirecting bookings directly to rail ticketing systems.
