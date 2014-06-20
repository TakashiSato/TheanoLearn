"""Base classes for all estimators."""
import inspect
import warnings

import numpy as np

import theano
import logging
import os
import datetime
import cPickle as pickle
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

###############################################################################
def iteritems(d, **kw):
    """Return an iterator over the (key, value) pairs of a dictionary."""
    return iter(getattr(d, "iteritems")(**kw))

###############################################################################
def _pprint(params, offset=0, printer=repr):
    """Pretty print the dictionary 'params'

    Parameters
    ----------
    params: dict
        The dictionary to pretty print

    offset: int
        The offset in characters to add at the begin of each line.

    printer:
        The function to convert entries to strings, typically
        the builtin str or repr

    """
    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i, (k, v) in enumerate(sorted(iteritems(params))):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if (this_line_length + len(this_repr) >= 75 or '\n' in this_repr):
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines


###############################################################################
class BaseEstimator(object):
    """Base class for all estimators in scikit-learn

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        args, varargs, kw, default = inspect.getargspec(init)
        if varargs is not None:
            raise RuntimeError("scikit-learn estimators should always "
                               "specify their parameters in the signature"
                               " of their __init__ (no varargs)."
                               " %s doesn't follow this convention."
                               % (cls, ))
        # Remove 'self'
        # XXX: This is going to fail if the init is a staticmethod, but
        # who would do this?
        args.pop(0)
        args.sort()
        return args

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The former have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in iteritems(params):
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if not name in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if not key in valid_params:
                    raise ValueError('Invalid parameter %s ' 'for estimator %s'
                                     % (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
                                               offset=len(class_name),),)
        

    def shared_dataset(self, data_xy):
        """ Load the dataset into shared variables """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX))
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX))
        return shared_x, shared_y

    def __getstate__(self):
        """ Return state sequence."""
        params = self.get_params()  # parameters set in constructor
        weights = [p.get_value() for p in self.estimator.params]
        state = (params, weights)
        return state

    def _set_weights(self, weights):
        """ Set fittable parameters from weights sequence."""
        i = iter(weights)

        for param in self.estimator.params:
            param.set_value(i.next())

    def __setstate__(self, state):
        """ Set parameters from state sequence."""
        params, weights = state
        self.set_params(**params)
        self.ready()
        self._set_weights(weights)

    def save(self, fpath='./models', fname=None, save_errorlog=False):
        """ Save a pickled representation of Model state. """
        if fname is None:
            # Generate filename based on date
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime('%Y%m%d-%H%M%S')
            class_name = self.__class__.__name__
            fname = '%s.%s' % (class_name, date_str)

        # Make directory if not exist save dir
        if os.path.isdir(fpath) is False:
            os.makedirs(fpath)

        fabspath = os.path.join(fpath, fname + '.pkl')

        logger.info("Saving to %s ..." % fabspath)
        file = open(fabspath, 'wb')
        state = self.__getstate__()
        pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()
    
        if (self.errorlog is not None) and (save_errorlog is True):
            fabspath = os.path.join(fpath, fname + '_Errorlog.pkl')
            logger.info("Saving to %s ..." % fabspath)
            file = open(fabspath, 'wb')
            pickle.dump(self.errorlog, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.close()
            self.save_errorlog_png(fpath=fpath, fname=fname)

    def load(self, path):
        """ Load model parameters from path. """
        logger.info("Loading from %s ..." % path)
        file = open(path, 'rb')
        state = pickle.load(file)
        self.__setstate__(state)
        file.close()

    def save_errorlog_png(self, fpath='./errorlog', fname=None):
        """ Save error logging graph """
        if fname is None:
            # Generate filename based on date
            class_name = self.__class__.__name__
            fname = '%s.%s' % (class_name, self.timestamp)

        # Make directory if not exist save dir
        if os.path.isdir(fpath) is False:
            os.makedirs(fpath)

        fabspath = os.path.join(fpath, 'Errorlog_' + fname + '.png')

        logger.info("Saving to %s ..." % fabspath)
        plt.close('all')
        plt.plot(self.errorlog[:,0],self.errorlog[:,1:])
        plt.yscale('log')
        plt.savefig(fabspath)

    def optional_output(self, train_set_x, show_norms=True, show_output=True):
        """ Produces some debugging output. """
        if show_norms:
            norm_output = []
            for param in self.estimator.params:
                norm_output.append('%s: %6.4f' % (param.name,
                                                   self.get_norms[param]()))
            logger.info("norms: {" + ', '.join(norm_output) + "}")

        if show_output:
            output_fn = self.predict
            logger.info("sample output: " + \
                    str(output_fn([train_set_x.get_value(
                        borrow=True)[0]]).flatten()))
