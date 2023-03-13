from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.notebook import tqdm
import numpy as np

from src.optimal_number import optimal_params
from src.main import check_similarity, conf_intervals


def final_guess(x, y, sigma, params, params_std, conf_level=2):
    aics, aic_probs, bics, bic_probs, m_aic, m_bic, cons_idx = optimal_params(x, y, params, sigma)

    params_opt = params[cons_idx]
    params_opt_std = params_std[cons_idx]
    indx = cons_idx

    intervals = conf_intervals(params_opt, params_opt_std, conf_level)
    check = check_similarity(params_opt, intervals)
    while check and indx > 1:
        indx = indx - 1
        params_opt = params[indx]
        params_opt_std = params_std[indx]
        intervals = conf_intervals(params_opt, params_opt_std, conf_level)
        check = check_similarity(params_opt, intervals)
    return indx, params_opt, params_std


def bootstrap(function, x, y_model, sigma, num=100, *args, **kwargs):
    sigma = np.exp(-5.014 * x - 4.099) / 0.022 * sigma

    with ProcessPoolExecutor(max_workers=10) as pool:
        samples = list(map(lambda _: (x, y_model + np.random.normal(0, sigma, len(x))), range(num)))

        futures = [pool.submit(function, x1, y1, *args, **kwargs) for x1, y1 in samples]

        results = []
        for future in tqdm(as_completed(futures), total=num):
            results.append(future.result())
        results = np.vstack(results)
        params_std = ((results - results.mean(0)) ** 2).mean(0) ** 0.5

        optim_futures = [pool.submit(final_guess, x1, y1, sigma, params, params_std)
                  for (x1, y1), params in zip(samples, results)]
        optim_results = []
        for future in as_completed(optim_futures):
            optim_results.append(future.result())
        optim_results = np.vstack(optim_results)

        return results, optim_results
