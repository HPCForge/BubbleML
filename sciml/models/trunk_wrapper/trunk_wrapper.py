
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
            use_bias = True,
            func_number = 1
            ):
            super().__init__()
            self.func_number = func_number
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
        
        for j in range(1, self.func_number+1):
            module_list['Tr{}LinM1'.format(j)] = torch.nn.Linear(1, self.width//self.func_number)
            module_list['Tr{}ActM1'.format(j)] = torch.nn.ReLU()
            for i in range(2, self.trunk_depth):
                module_list['Tr{}LinM{}'.format(j, i)] = torch.nn.Linear(self.width//self.func_number, self.width//self.func_number)
                module_list['Tr{}ActM{}'.format(j, i)] = torch.nn.ReLU()

        
        return module_list
    
    def __init_params(self):
        params = torch.nn.ParameterDict()
        params['bias'] = torch.nn.Parameter(torch.zeros([self.func_number]))
        return params

    
    def __initialize(self):
        for j in range(1, 1, self.func_number+1):
            for i in range(1, self.trunk_depth):
                torch.nn.init.kaiming_normal_(self.module_list['Tr{}LinM{}'.format(j, i)].weight)
                torch.nn.init.constant_(self.module_list['Tr{}LinM{}'.format(j, i)].bias, 0)

    def forward(self, x_branch, query_vector):
        # tenor.tensor(i for i in range(past_window))
        if self.training:
            x_branch = self.module_list["Branch"](x_branch[::self.fw, ...])
            x_branch = torch.repeat_interleave(x_branch, self.fw, dim=0)
        else:
            x_branch = self.module_list["Branch"](x_branch)

        query_vector = [query_vector.reshape(1, -1).t().reshape(-1, 1)]*self.func_number

        for j in range(self.func_number):
            for i in range(1, self.trunk_depth): 
                query_vector[j] = self.module_list[f'Tr{j+1}LinM{i}'](query_vector[j])
                query_vector[j] = self.module_list[f'Tr{j+1}ActM{i}'](query_vector[j])

        shape = (*x_branch.shape[:-3], self.out_channels//self.func_number, *x_branch.shape[-2:])
        query_vector = torch.cat([torch.reshape(vec, shape) for vec in query_vector], dim = -3)

        out = []
        for i in range(self.func_number):
            start = i*(self.out_channels//self.func_number)
            end = (i+1)*(self.out_channels//self.func_number)
            out += [torch.sum(x_branch[:,start:end] * query_vector[:,start:end], dim=1, keepdim=True) + self.params['bias'][i]]

        out = torch.cat(out, dim = -3)
        return out

    






