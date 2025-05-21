// 6 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
using SurfaceSeek;

NeuronLayer[] network = 
    { 
        new NeuronLayer(2, 3), 
        new Tanh(), 
        new NeuronLayer(3, 1), 
        new Tanh() 
    };
