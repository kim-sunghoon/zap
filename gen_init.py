import os

import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--planes',
            type = int,
            default=256
            )
    return parser.parse_args()

def init_gen(filename, planes):
    with open(filename, 'w') as f:
        f.write("        elif (cfg.filter_mode=\'x\' and self.planes={}):\n".format(planes))
        for i in range(0, planes):
            f.write("            self.weight1_{} = torch.empty((1,1,3,3), requires_grad = True).to(\"cuda\")\n".format(i+1))

        f.write("\n")

        for i in range(0, planes):
            f.write("            nn.init.kaiming_normal_(self.weight1_{}, mode=\'fan_out\', nonlinearity=\'relu\')\n".format(i+1))

        f.write("\n")

        for i in range(0, planes):
            f.write("            self.weight2_{} = torch.empty((1,1,3,3), requires_grad = True).to(\"cuda\")\n".format(i+1))

        f.write("\n")
        for i in range(0, planes):
            f.write("            nn.init.kaiming_normal_(self.weight2_{:3d}, mode=\'fan_out\', nonlinearity=\'relu\')\n".format(i+1))



def forward_gen(filename, planes):
    with open(filename, 'a') as f:
        f.write("    ############## forward ############\n")
        f.write("        elif (cfg.filter_mode=\'x\' and self.planes={}):\n".format(planes))
        f.write("            out1 = None\n")
        f.write("            out2 = None\n")
        f.write("\n")
        #  for i in range(0, planes):
        #      f.write("            out1_{} = None\n".format(i+1))
        #  f.write("\n")
        #  for i in range(0, planes):
        #      f.write("            out2_{} = None\n".format(i+1))
        #  f.write("\n")
        for i in range(0, planes):
            f.write("            out1_{} = F.conv2d(x[:, {}:{},:,:], self.weight1_{}, stride=1, padding=1, groups=1)\n".format(i+1, i, i+1, i+1))
        f.write("\n")
        f.write("            out1 = torch.cat[")
        for i in range(0, planes):
            if i+1 is not planes:
                f.write("out1_{}, ".format(i+1))
            #  elif i+1 % 10 == 0:
            #      f.write("\n")
            else:
                f.write("out1_{}], dim=1)\n ".format(i+1))

        f.write("\n")
        f.write("            out1 = self.bn1(out1)\n")
        f.write("            out1 = F.relu(out1)\n")
        for i in range(0, planes):
            f.write("            out2_{} = F.conv2d(out1[:, {}:{},:,:], self.weight2_{}, stride=1, padding=1, groups=1)\n".format(i+1, i, i+1, i+1))

        f.write("\n")
        f.write("            out2 = torch.cat[")
        for i in range(0, planes):
            if i+1 is not planes:
                f.write("out2_{}, ".format(i+1))
            #  elif i+1 % 10 == 0:
            #      f.write("\n")
            else:
                f.write("out2_{}], dim=1)\n ".format(i+1))
        f.write("            x_pred_mask = self.bn2(out2)\n")

if __name__ == "__main__":
    args = parse_opt()

    filename = "{}_{}.py".format('test', args.planes)
    init_gen(filename, args.planes)

    forward_gen(filename, args.planes)


