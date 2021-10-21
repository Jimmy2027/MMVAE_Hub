from torch.distributions import Normal, Laplace


def get_distr(distr_str: str):
    if distr_str.lower() == 'normal':
        return Normal
    elif distr_str.lower() == 'laplace':
        return Laplace
    else:
        raise ValueError(f'not implemented for distr_str {distr_str}')
