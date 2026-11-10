import numpy as np

class Runner:
    """Reuse input/output allocations on CPU, including input-copy cost in timings."""
    def __init__(self,session,mode):
        self.session=session;self.mode=mode
        self.input_name=session.get_inputs()[0].name;self.output_name=session.get_outputs()[0].name
        assert session.get_inputs()[0].shape==[1,3,224,224]
        assert session.get_outputs()[0].shape==[1,6]
        if mode=='buffer':
            self.input=np.empty((1,3,224,224),dtype=np.float32)
            self.output=np.empty((1,6),dtype=np.float32)
            self.binding=session.io_binding()
            self.binding.bind_cpu_input(self.input_name,self.input)
            self.binding.bind_output(self.output_name,'cpu',0,np.float32,self.output.shape,self.output.ctypes.data)
    def __call__(self,x):
        if self.mode=='baseline':return self.session.run(None,{self.input_name:x})[0]
        np.copyto(self.input,x)
        self.session.run_with_iobinding(self.binding)
        return self.output
