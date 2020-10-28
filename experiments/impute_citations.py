#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional
import torch.utils.data as data
import numpy as np
import sys

import sys
sys.path.append('.')
import scnn.scnn
import scnn.chebyshev


class MySCNN(nn.Module):
    def __init__(self, colors = 1):
        super().__init__()

        assert(colors > 0)
        self.colors = colors

        num_filters = 30 #20
        variance = 0.01 #0.001

        # Degree 0 convolutions.
        self.C0_1 = scnn.scnn.SimplicialConvolution(5, self.colors, num_filters*self.colors, variance=variance)
        self.C0_2 = scnn.scnn.SimplicialConvolution(5, num_filters*self.colors, num_filters*self.colors, variance=variance)
        self.C0_3 = scnn.scnn.SimplicialConvolution(5, num_filters*self.colors, self.colors, variance=variance)

        # Degree 1 convolutions.
        self.C1_1 = scnn.scnn.SimplicialConvolution(5, self.colors, num_filters*self.colors, variance=variance)
        self.C1_2 = scnn.scnn.SimplicialConvolution(5, num_filters*self.colors, num_filters*self.colors, variance=variance)
        self.C1_3 = scnn.scnn.SimplicialConvolution(5, num_filters*self.colors, self.colors, variance=variance)

        # Degree 2 convolutions.
        self.C2_1 = scnn.scnn.SimplicialConvolution(5, self.colors, num_filters*self.colors, variance=variance)
        self.C2_2 = scnn.scnn.SimplicialConvolution(5, num_filters*self.colors, num_filters*self.colors, variance=variance)
        self.C2_3 = scnn.scnn.SimplicialConvolution(5, num_filters*self.colors, self.colors, variance=variance)



    def forward(self, Ls, Ds, adDs, xs):
        assert(len(xs) == 3) # The three degrees are fed together as a list.

        assert(len(Ls) == len(Ds))
        Ms = [L.shape[0] for L in Ls]
        Ns = [D.shape[0] for D in Ds]

        Bs = [x.shape[0] for x in xs]
        C_ins = [x.shape[1] for x in xs]
        Ms = [x.shape[2] for x in xs]

        assert(Ms == [D.shape[1] for D in Ds])
        assert(Ms == [L.shape[1] for L in Ls])
        assert([adD.shape[0] for adD in adDs] == [D.shape[1] for D in Ds])
        assert([adD.shape[1] for adD in adDs] == [D.shape[0] for D in Ds])

        assert(Bs == len(Bs)*[Bs[0]])
        assert(C_ins == len(C_ins)*[C_ins[0]])

        out0_1 = self.C0_1(Ls[0], xs[0]) #+ self.D10_1(xs[1])
        out1_1 = self.C1_1(Ls[1], xs[1]) #+ self.D01_1(xs[0]) + self.D21_1(xs[2])
        out2_1 = self.C2_1(Ls[2], xs[2]) #+ self.D12_1(xs[1])

        out0_2 = self.C0_2(Ls[0], nn.LeakyReLU()(out0_1)) #+ self.D10_2(nn.LeakyReLU()(out1_1))
        out1_2 = self.C1_2(Ls[1], nn.LeakyReLU()(out1_1)) #+ self.D01_2(nn.LeakyReLU()(out0_1)) + self.D21_2(nn.LeakyReLU()(out2_1))
        out2_2 = self.C2_2(Ls[2], nn.LeakyReLU()(out2_1)) #+ self.D12_2(nn.LeakyReLU()(out1_1))

        out0_3 = self.C0_3(Ls[0], nn.LeakyReLU()(out0_2)) #+ self.D10_3(nn.LeakyReLU()(out1_2))
        out1_3 = self.C1_3(Ls[1], nn.LeakyReLU()(out1_2)) #+ self.D01_3(nn.LeakyReLU()(out0_2)) + self.D21_2(nn.LeakyReLU()(out2_2))
        out2_3 = self.C2_3(Ls[2], nn.LeakyReLU()(out2_2)) #+ self.D12_3(nn.LeakyReLU()(out1_2))

        #return [out0_3, torch.zeros_like(xs[1]), torch.zeros_like(xs[2])]
        #return [torch.zeros_like(xs[0]), out1_3, torch.zeros_like(xs[2])]
        return [out0_3, out1_3, out2_3]

def main():
    torch.manual_seed(1337)
    np.random.seed(1337)


    prefix = sys.argv[1] ##input

    logdir = sys.argv[2] ##output
    starting_node=sys.argv[3]
    percentage_missing_values=sys.argv[4]
    cuda = False

    topdim = 2


    laplacians = np.load('{}/{}_laplacians.npy'.format(prefix,starting_node),allow_pickle=True)
    boundaries = np.load('{}/{}_boundaries.npy'.format(prefix,starting_node),allow_pickle=True)



    Ls =[scnn.scnn.coo2tensor(scnn.chebyshev.normalize(laplacians[i],half_interval=True)) for i in range(topdim+1)] #####scnn.chebyshev.normalize ?
    Ds=[scnn.scnn.coo2tensor(boundaries[i].transpose()) for i in range(topdim+1)]
    adDs=[scnn.scnn.coo2tensor(boundaries[i]) for i in range(topdim+1)]


    network = MySCNN(colors = 1)


    learning_rate = 0.001
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.L1Loss(reduction="sum")
    #criterion = nn.MSELoss(reduction="sum")

    batch_size = 1

    num_params = 0
    print("Parameter counts:")
    for param in network.parameters():
        p = np.array(param.shape, dtype=int).prod()
        print(p)
        num_params += p
    print("Total number of parameters: %d" %(num_params))


    masks_all_deg = np.load('{}/{}_percentage_{}_known_values.npy'.format(prefix,starting_node,percentage_missing_values),allow_pickle=True) ## positive mask= indices that we keep ##1 mask #entries 0 degree
    masks=[list(masks_all_deg[i].values()) for i in range(len(masks_all_deg))]

    losslogf = open("%s/loss.txt" %(logdir), "w")

    cochain_target_alldegs = []
    signal = np.load('{}/{}_cochains.npy'.format(prefix,starting_node),allow_pickle=True)
    raw_data=[list(signal[i].values()) for i in range(len(signal))]
    for d in range(0, topdim+1):
        cochain_target = torch.zeros((batch_size, 1, len(raw_data[d])), dtype=torch.float, requires_grad = False)
        for i in range(0, batch_size):
            cochain_target[i, 0, :] = torch.tensor(raw_data[d], dtype=torch.float, requires_grad = False)

        cochain_target_alldegs.append(cochain_target)

    cochain_input_alldegs = []
    signal = np.load('{}/{}_percentage_{}_input_damaged.npy'.format(prefix,starting_node,percentage_missing_values),allow_pickle=True)
    raw_data=[list(signal[i].values()) for i in range(len(signal))]
    for d in range(0, topdim+1):

        cochain_input = torch.zeros((batch_size, 1, len(raw_data[d])), dtype=torch.float, requires_grad = False)

        for i in range(0, batch_size):
            cochain_input[i, 0, :] = torch.tensor(raw_data[d], dtype=torch.float, requires_grad = False)

        cochain_input_alldegs.append(cochain_input)

    #cochain_target_alldegs[0] = torch.zeros_like(cochain_target_alldegs[0])
    #cochain_target_alldegs[2] = torch.zeros_like(cochain_target_alldegs[2])

    #cochain_input_alldegs[0] = torch.zeros_like(cochain_input_alldegs[0])
    #cochain_input_alldegs[2] = torch.zeros_like(cochain_input_alldegs[2])

    print([float(len(masks[d]))/float(len(cochain_target_alldegs[d][0,0,:])) for d in range(0,2+1)])

    for i in range(0, 1000):
        xs = [cochain_input.clone() for cochain_input in cochain_input_alldegs]

        optimizer.zero_grad()
        ys = network(Ls, Ds, adDs, xs)

        loss = torch.FloatTensor([0.0])
        for b in range(0, batch_size):
            for d in range(0, topdim+1):
                loss += criterion(ys[d][b, 0, masks[d]], cochain_target_alldegs[d][b, 0, masks[d]])

        detached_ys = [ys[d].detach() for d in range(0, topdim+1)]

        if np.mod(i, 10) == 0:
            for d in range(0,topdim+1):
                np.savetxt("%s/output_%d_%d.txt" %(logdir, i, d), detached_ys[d][0,0,:])

        for d in range(0, topdim+1):
            predictionlogf = open("%s/prediction_%d_%d.txt" %(logdir, i, d), "w")
            actuallogf = open("%s/actual_%d_%d.txt" %(logdir, i, d), "w")

            for b in range(0, batch_size):
                for y in detached_ys[d][b, 0, masks[d]]:
                    predictionlogf.write("%f " %(y))
                predictionlogf.write("\n")
                for x in cochain_target_alldegs[d][b, 0, masks[d]]:
                    actuallogf.write("%f " %(x))
                actuallogf.write("\n")
            predictionlogf.close()
            actuallogf.close()


        losslogf.write("%d %f\n" %(i, loss.item()))
        losslogf.flush()

        loss.backward()
        optimizer.step()

    losslogf.close()

    name_networks=['C0_1,C0_2','C0_3','C1_1,C1_2','C1_3', 'C2_1,C2_2','C2_3']



if __name__ == "__main__":
    main()
