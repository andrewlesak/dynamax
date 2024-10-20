import typing

import tqdm.auto
import tqdm.notebook
import tqdm.std

import jax
from jax.debug import callback
from IPython import get_ipython


def while_tqdm(
    n: int,
    print_rate: int = 1,
    bar_id: int = 0,
    tqdm_type: str = "auto",
    disable: bool = False,
    **kwargs,
) -> typing.Callable:
    """
    tqdm progress bar for a JAX while loop. Allows for early 
    stopping of progress if some convergence criterion is met.

    Parameters
    ----------
    n : int
        Maximum number of iterations.
    print_rate: int
        Integer rate at which the progress bar will be updated,
        by default the print rate updates every step.
    bar_id: int
        Index used to properly reference each progress bar.
    tqdm_type: str
        Type of progress-bar, should be one of "auto", "std", or "notebook".
    disable: bool
        Optionally disable the progress bar.
    **kwargs
        Extra keyword arguments to pass to tqdm.

    Returns
    -------
    typing.Callable:
        Progress bar wrapping `body_fun` of `jax.lax.while_loop`
    """

    _update_progress_bar, _final_update = build_while_tqdm(n, print_rate, bar_id, tqdm_type, disable, **kwargs)

    def _bounded_tqdm(func):
        """Decorator that adds a tqdm progress bar to `body_fun` used in 
        `jax.lax.while_loop`. Note that the current iteration `iter_num` 
        and the `converged` bool must be the first and second elements 
        in the `carry` tuple, respectively. 
        """
        def wrapper_progress_bar(carry):
            iter_num, converged, *_ = carry
            _update_progress_bar(iter_num, converged, bar_id=bar_id)
            return func(carry)

        # Since `jax.lax.while_loop` will terminate before passing the last
        # iter to update, we have to close the pbars outside of the loop. 
        # This is achieved by attaching `final_update` to the `body_fun`
        def final_update(iter_num, converged, bar_id=bar_id):
            callback(_final_update, iter_num, converged, bar_id, ordered=True)

        wrapper_progress_bar.final_update = jax.tree_util.Partial(final_update)

        return wrapper_progress_bar

    return _bounded_tqdm


def _is_jupyter():
    """check if in IPython enviornment"""
    try:
        ipython = get_ipython()
        if 'ipykernel' in str(type(ipython)):
            return True
    except NameError:
        return False
    return False


def build_while_tqdm(
    n: int,
    print_rate: int,
    bar_id: int,
    tqdm_type: str,
    disable: bool,
    **kwargs,
) -> typing.Tuple[typing.Callable, typing.Callable]:
    """
    Build the tqdm progress bar on the host
    """

    if tqdm_type not in ("auto", "std", "notebook"):
        raise ValueError(
            'tqdm_type should be one of "auto", "std", or "notebook" '
            f'but got "{tqdm_type}"'
        )
    pbar = getattr(tqdm, tqdm_type).tqdm

    desc = kwargs.pop("desc", f"Running for {n:,} iterations")
    message = kwargs.pop("message", desc)
    position_offset = kwargs.pop("position", 0)

    for kwarg in ("total", "mininterval", "maxinterval", "miniters"):
        kwargs.pop(kwarg, None)

    if print_rate < 1:
        raise ValueError(f"Print rate should be > 0 got {print_rate}")
    elif print_rate > n:
        raise ValueError(
            "Print rate should be less than the "
            f"number of steps {n}, got {print_rate}"
        )

    def _check_valid_id(bar_id: int):
        """If traced, we must check the id with a callback"""
        if bar_id < 0:
            raise ValueError(f"Progress bar index should be >= 0, got {bar_id}")

    callback(_check_valid_id, bar_id, ordered=True)
    
    tqdm_bars = dict()

    # immediately initialize progress bars
    def _define_tqdm(bar_id):
        bar_id = int(bar_id)
        tqdm_bars[bar_id] = pbar(range(n), position=bar_id + position_offset, disable=disable, **kwargs)
        tqdm_bars[bar_id].set_description(message)
    
    callback(_define_tqdm, bar_id, ordered=True)

    def _update_tqdm(iter_num, converged, print_rate, bar_id: int):
        """Update progress bars"""
        bar_id = int(bar_id)

        if not converged:
            if (iter_num % print_rate == 0):
                tqdm_bars[bar_id].update(int(print_rate))
        else:
            # update if converged. this is required to properly terminate
            # progress bars that finish before others while vmapped.
            diff = iter_num - tqdm_bars[bar_id].n
            tqdm_bars[bar_id].update(int(diff))
            tqdm_bars[bar_id].clear()
            
            # close pbar to signify loop has converged only if in ipython env.
            # calling close() at this point messes up pbars in console.
            if _is_jupyter():
                tqdm_bars[bar_id].close()

    def _update_progress_bar(iter_num, converged, bar_id: int = 0):
        """
        Updates tqdm from a `jax.lax.while_loop`. Note that we cannot use lax.cond()
        for control flow if we also use vmap since vmap-of-cond executes both branch
        conditions. Instead, we callback a single update function that handles each case.
        """
        callback(_update_tqdm, iter_num, converged, print_rate, bar_id, ordered=True)

    def _final_update(final_iter, converged, bar_id: int = 0):
        """Final update for the progress bar"""
        bar_id = int(bar_id)
        diff = final_iter - tqdm_bars[bar_id].n
        if (final_iter == n) | converged:
            tqdm_bars[bar_id].update(int(diff))
            # clear() before close() to cleanly terminate pbars in console
            tqdm_bars[bar_id].clear() 
            tqdm_bars[bar_id].close()

    return _update_progress_bar, _final_update







def while_tqdm2(
    n: int,
    print_rate: int = 1,
    bar_id: int = 0,
    tqdm_type: str = "auto",
    disable: bool = False,
    **kwargs,
) -> typing.Callable:
    """
    tqdm progress bar for a JAX while loop. Allows for early 
    stopping of progress if some convergence criterion is met.

    Parameters
    ----------
    n : int
        Maximum number of iterations.
    print_rate: int
        Integer rate at which the progress bar will be updated,
        by default the print rate updates every step.
    bar_id: int
        Index used to properly reference each progress bar.
    tqdm_type: str
        Type of progress-bar, should be one of "auto", "std", or "notebook".
    disable: bool
        Optionally disable the progress bar.
    **kwargs
        Extra keyword arguments to pass to tqdm.

    Returns
    -------
    typing.Callable:
        Progress bar wrapping `body_fun` of `jax.lax.while_loop`
    """

    update_progress_bar, close_tqdm = build_while_tqdm2(n, print_rate, bar_id, tqdm_type, disable, **kwargs)

    def _bounded_tqdm(func):
        """Decorator that adds a tqdm progress bar to `body_fun` used in 
        `jax.lax.while_loop`. Note that the current iteration `iter_num` 
        and the `converged` bool must be the first and second elements 
        in the `carry` tuple, respectively. 
        """
        def wrapper_progress_bar(carry):
            # iter_num, converged, *_ = carry
            carry = update_progress_bar(carry, carry[0], carry[1], bar_id=bar_id)
            result = func(carry)
            # return close_tqdm(result, iter_num, converged)
            return close_tqdm(result, carry[0], carry[1])

        # # Since `jax.lax.while_loop` will terminate before passing the last
        # # iter to update, we have to close the pbars outside of the loop. 
        # # This is achieved by attaching `final_update` to the `body_fun`
        # def final_update(iter_num, converged, bar_id=bar_id):
        #     callback(final_update, iter_num, converged, bar_id, ordered=True)

        # wrapper_progress_bar.final_update = jax.tree_util.Partial(final_update)

        return wrapper_progress_bar

    return _bounded_tqdm


def build_while_tqdm2(
    n: int,
    print_rate: int,
    bar_id: int,
    tqdm_type: str,
    disable: bool,
    **kwargs,
) -> typing.Tuple[typing.Callable, typing.Callable]:
    """
    Build the tqdm progress bar on the host
    """

    if tqdm_type not in ("auto", "std", "notebook"):
        raise ValueError(
            'tqdm_type should be one of "auto", "std", or "notebook" '
            f'but got "{tqdm_type}"'
        )
    pbar = getattr(tqdm, tqdm_type).tqdm

    desc = kwargs.pop("desc", f"Running for {n:,} iterations")
    message = kwargs.pop("message", desc)
    position_offset = kwargs.pop("position", 0)

    for kwarg in ("total", "mininterval", "maxinterval", "miniters"):
        kwargs.pop(kwarg, None)

    if print_rate < 1:
        raise ValueError(f"Print rate should be > 0 got {print_rate}")
    elif print_rate > n:
        raise ValueError(
            "Print rate should be less than the "
            f"number of steps {n}, got {print_rate}"
        )

    def _check_valid_id(bar_id: int):
        """If traced, we must check the id with a callback"""
        if bar_id < 0:
            raise ValueError(f"Progress bar index should be >= 0, got {bar_id}")

    callback(_check_valid_id, bar_id, ordered=True)
    
    tqdm_bars = dict()

    # # immediately initialize progress bars
    # def _define_tqdm(bar_id):
    #     bar_id = int(bar_id)
    #     tqdm_bars[bar_id] = pbar(range(n), position=bar_id + position_offset, disable=disable, **kwargs)
    #     tqdm_bars[bar_id].set_description("Compiling")
    
    # callback(_define_tqdm, bar_id, ordered=True)

    def _define_tqdm(bar_id: int):
        bar_id = int(bar_id)
        tqdm_bars[bar_id] = pbar(
            total=n,
            position=bar_id + position_offset,
            desc=message,
            **kwargs,
        )
    remainder = n % print_rate or print_rate

    # def _define_tqdm(bar_id: int):
    #     tqdm_bars[int(bar_id)].set_description(message)

    def _update_tqdm(bar_id: int):
        tqdm_bars[int(bar_id)].update(print_rate)

    def _close_tqdm(bar_id: int):
        _pbar = tqdm_bars.pop(int(bar_id))
        _pbar.update(remainder)
        _pbar.clear()
        _pbar.close()

    # def _update_tqdm(iter_num, converged, bar_id: int):
    #     """Update progress bars"""
    #     bar_id = int(bar_id)

    #     if not converged:
    #         if (iter_num % print_rate == 0):
    #             tqdm_bars[bar_id].update(int(print_rate))
    #     else:
    #         # update if converged. this is required to properly terminate
    #         # progress bars that finish before others while vmapped.
    #         diff = iter_num - tqdm_bars[bar_id].n
    #         tqdm_bars[bar_id].update(int(diff))
    #         tqdm_bars[bar_id].clear()
            
    #         # close pbar to signify loop has converged only if in ipython env.
    #         # calling close() at this point messes up pbars in console.
    #         if _is_jupyter():
    #             tqdm_bars[bar_id].close()

    def update_progress_bar(carry: typing.Any, iter_num: int, converged: bool, bar_id: int = 0):
        """
        Updates tqdm from a JAX while loop. We must pass `carry` through 
        the updates to properly order the callbacks.
        """

        def _inner_init(_iter_num, _converged, _carry):
            callback(_define_tqdm, bar_id, ordered=True)
            return _carry

        def _inner_update(i, _converged, _carry):
            # # callback(_update_tqdm, iter_num, converged, bar_id, ordered=True)
            # callback(_update_tqdm, carry[0], carry[1], bar_id, ordered=True)
            # return _carry
            _ = jax.lax.cond(
                i % print_rate == 0,
                lambda: callback(_update_tqdm, bar_id, ordered=True),
                lambda: None,
            )
            return _carry
        
        carry = jax.lax.cond(
            # iter_num == 0,
            carry[0] == 0,
            _inner_init,
            _inner_update,
            iter_num,
            converged,
            carry,
        )

        return carry

    def close_tqdm(result: typing.Any, iter_num: int, converged: bool, bar_id: int = 0):
        def _inner_close(_result):
            callback(_close_tqdm, bar_id, ordered=True)
            return _result

        result = jax.lax.cond(
            (iter_num + 1 == n) | converged, 
            _inner_close, 
            lambda r: r, result
        )
        return result
    
    # def _final_update(final_iter, converged, bar_id: int = 0):
    #     """Final update for the progress bar"""
    #     bar_id = int(bar_id)
    #     diff = final_iter - tqdm_bars[bar_id].n
    #     if (final_iter == n) | converged:
    #         tqdm_bars[bar_id].update(int(diff))
    #         # clear() before close() to cleanly terminate pbars in console
    #         tqdm_bars[bar_id].clear() 
    #         tqdm_bars[bar_id].close()

    return update_progress_bar, close_tqdm    
    # return _update_progress_bar, _final_update