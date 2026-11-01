"""Optional exact TreeSHAP accelerator; not required by the standard workflow.

FastTreeSHAP 0.1.6 needs legacy NumPy scalar aliases under NumPy 2. The aliases
below reproduce those type conversions; fitted models are never modified.
The reference shap implementation remains the default in interpretability.py.
"""
import numpy as np


def activate():
    import shap
    if getattr(shap.TreeExplainer,'_attrition_fast_backend',False):return
    def obj2sctype(obj,default=None):
        if isinstance(obj,type) and issubclass(obj,np.generic):return obj
        if isinstance(obj,np.ndarray):return obj.dtype.type
        try:return np.dtype(obj).type
        except (TypeError,ValueError):return default
    for name,value in {'obj2sctype':obj2sctype,'int':int,'float':float,'bool':bool,'object':object}.items():
        if name not in np.__dict__:setattr(np,name,value)
    import fasttreeshap
    class ExactFastTreeExplainer:
        _attrition_fast_backend=True
        def __init__(self,model,**kwargs):
            self.impl=fasttreeshap.TreeExplainer(model,algorithm='v1',n_jobs=1,**kwargs)
        @property
        def expected_value(self):return self.impl.expected_value
        def shap_values(self,X,**kwargs):
            values=self.impl.shap_values(X,**kwargs)
            return np.stack(values,axis=-1) if isinstance(values,list) else values
    shap.TreeExplainer=ExactFastTreeExplainer
