# Fictional grocery data card

- **Origin:** original seeded NumPy generator in `grocery_planner/data.py`; no external customer records or proprietary data.
- **Size:** 90 fictional days × 60 orders = 5,400 rows. IDs are synthetic. Three store codes describe anonymous simulated stores.
- **Features:** item count, produce fraction, frozen fraction, aisle count, congestion and store code. These are assumed available at the planning instant.
- **Target:** individual shopping minutes from a nonlinear equation plus Gaussian noise, clipped at two minutes. Distance is generated separately and used in the dispatch simulator, not in the shopping-duration model.
- **Split:** days 0–49 training (3,000), 50–64 calibration (900), 65–89 final prediction test (1,500). Model parameters and buffer rule are fixed in code.
- **Dispatch sample:** seed 703 chooses 24 orders without replacement per held-out day; independently generated remaining promise windows range from 25 to 68 minutes. Six shoppers, two at each store, have capacities of 43/57 items and pace factors 1.0/1.12. All inputs and realized labels are saved in `data/dispatch-waves.json`.
- **Trip approximation:** 3 fixed minutes + 0.9 × individual duration for singletons (0.68 × summed duration for multi-order batches), multiplied by shopper pace, + 1.4 minutes per summed kilometer + 1.2 minutes per order. Planning adds the calibrated trip reserve; realized evaluation does not. All orders in a batch finish at that batch's modeled completion time.
- **Reserve:** three times the calibration 90th-percentile positive individual residual, 4.96885 minutes in the reference run. It is a heuristic, not a theorem about joint trip coverage.
- **Interpretation:** zero observed late orders under the buffer does not guarantee zero future misses. Stationary synthetic data and known simulator structure make this a methods demonstration, not a retailer performance estimate.
- **License:** MIT with the repository. Fictional data may be redistributed under that license.
- **Privacy:** no addresses, shoppers, store identities, resume contact information or employer records.

Data, model, prediction and dispatch-input digests appear in `results/metrics.json`. Model output is JSON rather than an unsafe pickle. The fixture is reproducible from fixed seeds; local timing is expected to vary.
