import torch
import torch.nn as nn
import torch.nn.parallel

class DCGAN_D(nn.Container):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential(
            # input is nc x isize x isize
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        i, csize, cndf = 2, isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(str(i),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module(str(i+1),
                            nn.BatchNorm2d(cndf))
            main.add_module(str(i+2),
                            nn.LeakyReLU(0.2, inplace=True))
            i += 3

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module(str(i),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module(str(i+1),
                            nn.BatchNorm2d(out_feat))
            main.add_module(str(i+2),
                            nn.LeakyReLU(0.2, inplace=True))
            i+=3
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module(str(i),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        output = output.mean(0)
        return output.view(1)

class DCGAN_G(nn.Container):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(cngf),
            nn.ReLU(True),
        )

        i, csize, cndf = 3, 4, cngf
        while csize < isize//2:
            main.add_module(str(i),
                nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module(str(i+1),
                            nn.BatchNorm2d(cngf//2))
            main.add_module(str(i+2),
                            nn.ReLU(True))
            i += 3
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(str(i),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module(str(i+1),
                            nn.BatchNorm2d(cngf))
            main.add_module(str(i+2),
                            nn.ReLU(True))
            i += 3

        main.add_module(str(i),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module(str(i+1), nn.Tanh())
        self.main = main

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        return nn.parallel.data_parallel(self.main, input, gpu_ids)

###############################################################################
class DCGAN_D_nobn(nn.Container):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D_nobn, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential(
            # input is nc x isize x isize
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        i, csize, cndf = 2, isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(str(i),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module(str(i+1),
                            nn.LeakyReLU(0.2, inplace=True))
            i += 2

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module(str(i),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module(str(i+1),
                            nn.LeakyReLU(0.2, inplace=True))
            i+=2
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module(str(i),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        output = output.mean(0)
        return output.view(1)

class DCGAN_G_nobn(nn.Container):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G_nobn, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False),
            nn.ReLU(True),
        )

        i, csize, cndf = 3, 4, cngf
        while csize < isize//2:
            main.add_module(str(i),
                nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module(str(i+1),
                            nn.ReLU(True))
            i += 2
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(str(i),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module(str(i+1),
                            nn.ReLU(True))
            i += 2

        main.add_module(str(i),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module(str(i+1), nn.Tanh())
        self.main = main

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        return nn.parallel.data_parallel(self.main, input, gpu_ids)
