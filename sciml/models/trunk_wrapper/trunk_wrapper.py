
import torch
import yaml


class TrunkWrapper(torch.nn.Module):
    def __init__(self, 
            model, 
            forward_window,
            out_channels,
            domain_rows,
            domain_cols,
            trunk_depth,
            use_bias = True
            ):
            super().__init__()
            self.trunk_depth = trunk_depth
            self.fw = forward_window
            self.use_bias = use_bias
            self.out_channels = out_channels
            self.domain_rows = domain_rows
            self.domain_cols = domain_cols
            self.width = int(domain_cols * domain_rows * out_channels)


            self.model = model
            self.module_list = self.__init_module_list()
            self.params = self.__init_params()
            self.__initialize()



    def __init_module_list(self):
        module_list = torch.nn.ModuleDict()
        module_list["Branch"] = self.model
        module_list['TrLinM1'] = torch.nn.Linear(1, self.width)
        module_list['TrActM1'] = torch.nn.ReLU()
        for i in range(2, self.trunk_depth):
            module_list['TrLinM{}'.format(i)] = torch.nn.Linear(self.width, self.width)
            if (i == self.trunk_depth-1):
                module_list['TrActM{}'.format(i)] = torch.nn.Softmax(dim=-1)
            else:
                module_list['TrActM{}'.format(i)] = torch.nn.ReLU()

        
        return module_list
    
    def __init_params(self):
        params = torch.nn.ParameterDict()
        params['bias'] = torch.nn.Parameter(torch.zeros([1]))
        return params

    
    def __initialize(self):
        for i in range(1, self.trunk_depth):
            torch.nn.init.kaiming_normal_(self.module_list['TrLinM{}'.format(i)].weight)
            torch.nn.init.constant_(self.module_list['TrLinM{}'.format(i)].bias, 0)

    def forward(self, x_branch, query_vector):
        # tenor.tensor(i for i in range(past_window))
        if self.training:
            x_branch = self.module_list["Branch"](x_branch[::self.fw, ...])
            x_branch = torch.repeat_interleave(x_branch, self.fw, dim=0)
        else:
            x_branch = self.module_list["Branch"](x_branch)

        for i in range(1, self.trunk_depth): 
            query_vector = self.module_list[f'TrLinM{i}'](query_vector)
            query_vector = self.module_list[f'TrActM{i}'](query_vector)

        query_vector = torch.reshape(query_vector, x_branch.shape)
        
        return torch.sum(x_branch * query_vector, dim=1) + self.params['bias']

    






