
import torch
from models.get_model import get_model
import yaml


class trunk_wrapper(nn.Modules):
    def init(self, 
            model_name, 
            in_channels,
            out_channels,
            domain_rows,
            domain_cols,
            trunk_depth,
            exp,
            use_bias = True
            ):
            
            self.trunk_depth = trunk_depth
            self.use_bias = use_bias
            self.out_channels = out_channels
            self.domain_rows = domain_rows
            self.domain_cols = domain_cols
            self.width = domain_cols * domain_rows * out_channels


            self.model = get_model(model_name,
              in_channels,
              out_channels,
              domain_rows,
              domain_cols,
              exp)
            self.modules = self.__init_modules()
            self.params = self.__init_params()
            self.__initialize()



    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules["Branch"] = self.model
        modules['TrLinM1'] = torch.nn.Linear(1, self.width)
        modules['TrActM1'] = torch.nn.ReLU()
        for i in range(2, self.trunk_depth):
            modules['TrLinM{}'.format(i)] = torch.nn.Linear(self.width, self.width)
            if (i == self.trunk_depth-1):
                modules['TrActM{}'.format(i)] = torch.nn.Softmax(dim=-1)
            else:
                modules['TrActM{}'.format(i)] = torch.nn.ReLU()

        
        return modules
    
    def __init_params(self):
        params = torch.nn.ParameterDict()
        params['bias'] = torch.nn.Parameter(torch.zeros([1]))
        return params

    
    def __initialize(self):
        for i in range(1, self.trunk_depth):
            torch.nn.init.kaiming_normal_(self.modules['TrLinM{}'.format(i)].weight)
            torch.nn.init.constant_(self.modules['TrLinM{}'.format(i)].bias, 0)

    def forward(self, x_branch, query_vector):
        # tenor.tensor(i for i in range(past_window))
        x_branch = self.modules["Branch"](x_branch)

        for i in range(1, self.trunk_depth): 
            query_vector = self.modules[f'TrActM{i}'](query_vector)

        query_vector = torch.reshape(query_vector, x_branch.shape)
        
        return torch.sum(x_branch * query_vector, dim=1) + self.params['bias']

    






