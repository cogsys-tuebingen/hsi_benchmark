import copy

import torch
from torch import nn
import numpy as np
import re

#from classification.deep_learning_pretrain import * #_collate_fn, get_pretrain_data
#from classification.deep_learning import *
from classification.models.deephs_hyve_net import DeepHSNet_with_HyVEConv
from classification.models.resnet import BasicBlock, Bottleneck
from classification.models.resnet_hyve import ResnetHyve
from dataloader.basic_dataloader import get_channel_wavelengths
from dataloader.debris_dataloader import CLASS_LABEL_2_ID_MAPPING
from dataloader.hrss_dataloader import SCENE_2_LABEL_2_ID_MAPPING, Scene
from dataloader.valid_dataset_configs import VALID_DATASET_CONFIG
from evaluation import evaluate_predictions_on_test_set




class Multihead_DeepHSNet_with_HyVEConv(DeepHSNet_with_HyVEConv):
    def __init__(self, wavelength_range, num_of_wrois, enable_extension=True,
                 num_classes=3, stop_gaussian_gradient=False, num_channels=None):
        super(Multihead_DeepHSNet_with_HyVEConv, self).__init__(wavelength_range, num_of_wrois, enable_extension=True,
                                                                num_classes=num_classes, stop_gaussian_gradient=False,
                                                                num_channels=None)

        # FIXME: num_classes = max(num_classes) or individual num_classes per head??

        self.remotesensing_indianpines_fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[2]),
            #nn.Linear(self.hidden_layers[2], len(SCENE_2_LABEL_2_ID_MAPPING[Scene.INDIAN_PINES])),
            nn.Linear(self.hidden_layers[2], num_classes),
        )
        self.remotesensing_salinas_fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[2]),
            #nn.Linear(self.hidden_layers[2], len(SCENE_2_LABEL_2_ID_MAPPING[Scene.SALINAS])),
            nn.Linear(self.hidden_layers[2], num_classes),
        )
        self.remotesensing_paviau_fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[2]),
            #nn.Linear(self.hidden_layers[2], len(SCENE_2_LABEL_2_ID_MAPPING[Scene.PAVIA_UNIVERSITY])),
            nn.Linear(self.hidden_layers[2], num_classes),
        )

        # self.fruit_ripeness_fc = nn.Sequential(
        #     nn.Sigmoid(),
        #     nn.BatchNorm1d(self.hidden_layers[2]),
        #     #nn.Linear(self.hidden_layers[2], 3),
        #     nn.Linear(self.hidden_layers[2], num_classes),
        # )
        self.avocado_ripeness_fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[2]),
            # nn.Linear(self.hidden_layers[2], 3),
            nn.Linear(self.hidden_layers[2], num_classes),
        )
        self.kiwi_ripeness_fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[2]),
            # nn.Linear(self.hidden_layers[2], 3),
            nn.Linear(self.hidden_layers[2], num_classes),
        )
        self.mango_ripeness_fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[2]),
            # nn.Linear(self.hidden_layers[2], 3),
            nn.Linear(self.hidden_layers[2], num_classes),
        )
        self.kaki_ripeness_fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[2]),
            # nn.Linear(self.hidden_layers[2], 3),
            nn.Linear(self.hidden_layers[2], num_classes),
        )
        self.papaya_ripeness_fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[2]),
            # nn.Linear(self.hidden_layers[2], 3),
            nn.Linear(self.hidden_layers[2], num_classes),
        )
        # self.fruit_firmness_fc = nn.Sequential(
        #     nn.Sigmoid(),
        #     nn.BatchNorm1d(self.hidden_layers[2]),
        #     #nn.Linear(self.hidden_layers[2], 3),
        #     nn.Linear(self.hidden_layers[2], num_classes),
        # )
        self.avocado_firmness_fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[2]),
            # nn.Linear(self.hidden_layers[2], 3),
            nn.Linear(self.hidden_layers[2], num_classes),
        )
        self.kiwi_firmness_fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[2]),
            # nn.Linear(self.hidden_layers[2], 3),
            nn.Linear(self.hidden_layers[2], num_classes),
        )
        self.mango_firmness_fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[2]),
            # nn.Linear(self.hidden_layers[2], 3),
            nn.Linear(self.hidden_layers[2], num_classes),
        )
        self.kaki_firmness_fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[2]),
            # nn.Linear(self.hidden_layers[2], 3),
            nn.Linear(self.hidden_layers[2], num_classes),
        )
        self.papaya_firmness_fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[2]),
            # nn.Linear(self.hidden_layers[2], 3),
            nn.Linear(self.hidden_layers[2], num_classes),
        )
        # self.fruit_sugar_fc = nn.Sequential(
        #     nn.Sigmoid(),
        #     nn.BatchNorm1d(self.hidden_layers[2]),
        #     #nn.Linear(self.hidden_layers[2], 3),
        #     nn.Linear(self.hidden_layers[2], num_classes),
        # )
        self.kiwi_sugar_fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[2]),
            #nn.Linear(self.hidden_layers[2], 3),
            nn.Linear(self.hidden_layers[2], num_classes),
        )
        self.mango_sugar_fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[2]),
            #nn.Linear(self.hidden_layers[2], 3),
            nn.Linear(self.hidden_layers[2], num_classes),
        )
        self.kaki_sugar_fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[2]),
            #nn.Linear(self.hidden_layers[2], 3),
            nn.Linear(self.hidden_layers[2], num_classes),
        )
        self.papaya_sugar_fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[2]),
            #nn.Linear(self.hidden_layers[2], 3),
            nn.Linear(self.hidden_layers[2], num_classes),
        )

        self.debris_patchwise_fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[2]),
            #nn.Linear(self.hidden_layers[2], len(CLASS_LABEL_2_ID_MAPPING)),
            nn.Linear(self.hidden_layers[2], num_classes),
        )
        self.debris_objectwise_fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[2]),
            #nn.Linear(self.hidden_layers[2], len(CLASS_LABEL_2_ID_MAPPING)),
            nn.Linear(self.hidden_layers[2], num_classes),
        )

        # self.group_configs = ['remote_sensing/indian_pines/.*', 'remote_sensing/salinas/.*', 'remote_sensing/paviaU/.*',
        #                       'fruit/.*/ripeness/.*', 'fruit/.*/firmness/.*', 'fruit/.*/sugar/.*',
        #                       'debris/.*/patchwise', 'debris/.*/objectwise']
        self.group_configs = ['remote_sensing/indian_pines/.*', 'remote_sensing/salinas/.*', 'remote_sensing/paviaU/.*',
                              'fruit/avocado/ripeness/.*', 'fruit/kiwi/ripeness/.*', 'fruit/mango/ripeness/.*', 'fruit/kaki/ripeness/.*', 'fruit/papaya/ripeness/.*',
                              'fruit/avocado/firmness/.*', 'fruit/kiwi/firmness/.*', 'fruit/mango/firmness/.*', 'fruit/kaki/firmness/.*', 'fruit/papaya/firmness/.*',
                              'fruit/kiwi/sugar/.*', 'fruit/mango/sugar/.*', 'fruit/kaki/sugar/.*', 'fruit/papaya/sugar/.*',
                              'debris/.*/patchwise', 'debris/.*/objectwise']
        # self.group_heads = [self.remotesensing_indianpines_fc, self.remotesensing_salinas_fc, self.remotesensing_paviau_fc,
        #             self.fruit_ripeness_fc, self.fruit_firmness_fc, self.fruit_sugar_fc,
        #             self.debris_patchwise_fc, self.debris_objectwise_fc]
        self.group_heads = [self.remotesensing_indianpines_fc, self.remotesensing_salinas_fc, self.remotesensing_paviau_fc,
                            self.avocado_ripeness_fc, self.kiwi_ripeness_fc, self.mango_ripeness_fc, self.kaki_ripeness_fc, self.papaya_ripeness_fc,
                            self.avocado_firmness_fc, self.kiwi_firmness_fc, self.mango_firmness_fc, self.kaki_firmness_fc, self.papaya_firmness_fc,
                            self.kiwi_sugar_fc, self.mango_sugar_fc, self.kaki_sugar_fc, self.papaya_sugar_fc,
                            self.debris_patchwise_fc, self.debris_objectwise_fc]

        self.init_params()

    def forward(self, x, meta_data=None):
        assert meta_data is not None
        configs = np.array([m.config for m in meta_data])
        #print(configs, '\n')

        channel_wavelengths = get_channel_wavelengths(meta_data).type_as(x)

        out_conv = self.hyve_conv(x, channel_wavelengths=channel_wavelengths)
        out_conv = self.conv(out_conv)

        out_flat = out_conv.view(x.shape[0], -1)

        out_fc = []
        for c, fc in zip(self.group_configs, self.group_heads):
            o = out_flat[[re.search(c, ci) is not None for ci in configs]]
            if o.shape[0] > 0: # FIXME: avoid BatchNorm error for o.shape[0] = 1
                out_fc.append(fc(o))
        out_fc = torch.concatenate(out_fc)

        return out_fc

    def reset_head(self, seed=0, num_classes=None, BN=False):
        print('Reset head')
        if num_classes is None:
            #self.init_params(layers=self.fc, BN=BN, seed=seed)
            for h in self.group_heads:
                self.init_params(layers=h, BN=BN, seed=seed)
        else:
            print('-> {} classes'.format(num_classes))
            self.num_classes = num_classes

            # #self.fc = nn.Sequential(*list(self.fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            #
            # self.remotesensing_indianpines_fc = nn.Sequential(*list(self.remotesensing_indianpines_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            # self.remotesensing_salinas_fc = nn.Sequential(*list(self.remotesensing_salinas_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            # self.remotesensing_paviau_fc = nn.Sequential(*list(self.remotesensing_paviau_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            #
            # #self.fruit_ripeness_fc = nn.Sequential(*list(self.fruit_ripeness_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            # self.avocado_ripeness_fc = nn.Sequential(*list(self.avocado_ripeness_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            # self.kiwi_ripeness_fc = nn.Sequential(*list(self.kiwi_ripeness_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            # self.mango_ripeness_fc = nn.Sequential(*list(self.mango_ripeness_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            # self.kaki_ripeness_fc = nn.Sequential(*list(self.kaki_ripeness_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            # self.papaya_ripeness_fc = nn.Sequential(*list(self.papaya_ripeness_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            # #self.fruit_firmness_fc = nn.Sequential(*list(self.fruit_firmness_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            # self.avocado_firmness_fc = nn.Sequential(*list(self.avocado_firmness_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            # self.kiwi_firmness_fc = nn.Sequential(*list(self.kiwi_firmness_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            # self.mango_firmness_fc = nn.Sequential(*list(self.mango_firmness_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            # self.kaki_firmness_fc = nn.Sequential(*list(self.kaki_firmness_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            # self.papaya_firmness_fc = nn.Sequential(*list(self.papaya_firmness_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            # #self.fruit_sugar_fc = nn.Sequential(*list(self.fruit_sugar_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            # self.kiwi_sugar_fc = nn.Sequential(*list(self.kiwi_sugar_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            # self.mango_sugar_fc = nn.Sequential(*list(self.mango_sugar_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            # self.kaki_sugar_fc = nn.Sequential(*list(self.kaki_sugar_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            # self.papaya_sugar_fc = nn.Sequential(*list(self.papaya_sugar_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            #
            # self.debris_patchwise_fc = nn.Sequential(*list(self.debris_patchwise_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            # self.debris_objectwise_fc = nn.Sequential(*list(self.debris_objectwise_fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            #
            # # self.group_heads = [self.remotesensing_indianpines_fc, self.remotesensing_salinas_fc,
            # #                     self.remotesensing_paviau_fc,
            # #                     self.fruit_ripeness_fc, self.fruit_firmness_fc, self.fruit_sugar_fc,
            # #                     self.debris_patchwise_fc, self.debris_objectwise_fc]
            # self.group_heads = [self.remotesensing_indianpines_fc, self.remotesensing_salinas_fc, self.remotesensing_paviau_fc,
            #                     self.avocado_ripeness_fc, self.kiwi_ripeness_fc, self.mango_ripeness_fc, self.kaki_ripeness_fc, self.papaya_ripeness_fc,
            #                     self.avocado_firmness_fc, self.kiwi_firmness_fc, self.mango_firmness_fc, self.kaki_firmness_fc, self.papaya_firmness_fc,
            #                     self.kiwi_sugar_fc, self.mango_sugar_fc, self.kaki_sugar_fc, self.papaya_sugar_fc,
            #                     self.debris_patchwise_fc, self.debris_objectwise_fc]
            #
            # #self.init_params(layers=self.fc, BN=BN, seed=seed)
            # for h in self.group_heads:
            #     self.init_params(layers=h, BN=BN, seed=seed)

            if not BN:
                lin = nn.Linear(self.hidden_layers[-1], self.num_classes)
                # for n, p in lin.named_parameters():
                #     print(n, p)

                self.init_params(layers=nn.Sequential(lin), BN=False, seed=seed)
                # for n, p in lin.named_parameters():
                #     print(n, p)

                self.remotesensing_indianpines_fc = nn.Sequential(*list(self.remotesensing_indianpines_fc.children())[:-1] + [lin])
                self.remotesensing_salinas_fc = nn.Sequential(*list(self.remotesensing_salinas_fc.children())[:-1] + [lin])
                self.remotesensing_paviau_fc = nn.Sequential(*list(self.remotesensing_paviau_fc.children())[:-1] + [lin])

                self.avocado_ripeness_fc = nn.Sequential(*list(self.avocado_ripeness_fc.children())[:-1] + [lin])
                self.kiwi_ripeness_fc = nn.Sequential(*list(self.kiwi_ripeness_fc.children())[:-1] + [lin])
                self.mango_ripeness_fc = nn.Sequential(*list(self.mango_ripeness_fc.children())[:-1] + [lin])
                self.kaki_ripeness_fc = nn.Sequential(*list(self.kaki_ripeness_fc.children())[:-1] + [lin])
                self.papaya_ripeness_fc = nn.Sequential(*list(self.papaya_ripeness_fc.children())[:-1] + [lin])
                self.avocado_firmness_fc = nn.Sequential(*list(self.avocado_firmness_fc.children())[:-1] + [lin])
                self.kiwi_firmness_fc = nn.Sequential(*list(self.kiwi_firmness_fc.children())[:-1] + [lin])
                self.mango_firmness_fc = nn.Sequential(*list(self.mango_firmness_fc.children())[:-1] + [lin])
                self.kaki_firmness_fc = nn.Sequential(*list(self.kaki_firmness_fc.children())[:-1] + [lin])
                self.papaya_firmness_fc = nn.Sequential(*list(self.papaya_firmness_fc.children())[:-1] + [lin])
                self.kiwi_sugar_fc = nn.Sequential(*list(self.kiwi_sugar_fc.children())[:-1] + [lin])
                self.mango_sugar_fc = nn.Sequential(*list(self.mango_sugar_fc.children())[:-1] + [lin])
                self.kaki_sugar_fc = nn.Sequential(*list(self.kaki_sugar_fc.children())[:-1] + [lin])
                self.papaya_sugar_fc = nn.Sequential(*list(self.papaya_sugar_fc.children())[:-1] + [lin])

                self.debris_patchwise_fc = nn.Sequential(*list(self.debris_patchwise_fc.children())[:-1] + [lin])
                self.debris_objectwise_fc = nn.Sequential(*list(self.debris_objectwise_fc.children())[:-1] + [lin])

                self.group_heads = [self.remotesensing_indianpines_fc, self.remotesensing_salinas_fc,
                                    self.remotesensing_paviau_fc,
                                    self.avocado_ripeness_fc, self.kiwi_ripeness_fc, self.mango_ripeness_fc,
                                    self.kaki_ripeness_fc, self.papaya_ripeness_fc,
                                    self.avocado_firmness_fc, self.kiwi_firmness_fc, self.mango_firmness_fc,
                                    self.kaki_firmness_fc, self.papaya_firmness_fc,
                                    self.kiwi_sugar_fc, self.mango_sugar_fc, self.kaki_sugar_fc, self.papaya_sugar_fc,
                                    self.debris_patchwise_fc, self.debris_objectwise_fc]

            else:
                fc = nn.Sequential(
                    nn.Sigmoid(),
                    nn.BatchNorm1d(self.hidden_layers[-1]),
                    nn.Linear(self.hidden_layers[-1], self.num_classes),
                )
                # for n, p in fc.named_parameters():
                #     print(n, p)

                self.init_params(layers=fc, BN=True, seed=seed)
                # for n, p in fc.named_parameters():
                #     print(n, p)

                self.remotesensing_indianpines_fc, self.remotesensing_salinas_fc, self.remotesensing_paviau_fc, self.avocado_ripeness_fc, self.kiwi_ripeness_fc, self.mango_ripeness_fc, self.kaki_ripeness_fc, self.papaya_ripeness_fc, self.avocado_firmness_fc, self.kiwi_firmness_fc, self.mango_firmness_fc, self.kaki_firmness_fc, self.papaya_firmness_fc, self.kiwi_sugar_fc, self.mango_sugar_fc, self.kaki_sugar_fc, self.papaya_sugar_fc, self.debris_patchwise_fc, self.debris_objectwise_fc = (3+ 5 +5 +4 +2) * [copy.deepcopy(fc)]

                self.group_heads = [self.remotesensing_indianpines_fc, self.remotesensing_salinas_fc,
                                    self.remotesensing_paviau_fc,
                                    self.avocado_ripeness_fc, self.kiwi_ripeness_fc, self.mango_ripeness_fc,
                                    self.kaki_ripeness_fc, self.papaya_ripeness_fc,
                                    self.avocado_firmness_fc, self.kiwi_firmness_fc, self.mango_firmness_fc,
                                    self.kaki_firmness_fc, self.papaya_firmness_fc,
                                    self.kiwi_sugar_fc, self.mango_sugar_fc, self.kaki_sugar_fc, self.papaya_sugar_fc,
                                    self.debris_patchwise_fc, self.debris_objectwise_fc]

            # for n, p in self.group_heads[0].named_parameters():
            #     print(n, p)
            # for n, p in self.remotesensing_indianpines_fc.named_parameters():
            #     print(n, p)
            #
            # for n, p in self.group_heads[5].named_parameters():
            #     print(n, p)

class Multihead_ResNet_with_HyVEConv(ResnetHyve):
    def __init__(self, wavelength_range, num_of_wrois, enable_extension=True,
                 num_classes=3, stop_gaussian_gradient=False, num_channels=None,
                 arch: str = 'resnet18'):
        if arch == 'resnet18':
            block, layers = BasicBlock, [2, 2, 2, 2]
        elif arch == 'resnet152':
            block, layers = Bottleneck, [3, 8, 36, 3]
        super(Multihead_ResNet_with_HyVEConv, self).__init__(block, layers,
                                                             num_classes=num_classes,
                                                             wavelength_range=wavelength_range,
                                                             num_of_wrois=num_of_wrois)

        # FIXME: num_classes = max(num_classes) or individual num_classes per head??

        self.remotesensing_indianpines_fc = nn.Linear(512 * self.block.expansion, num_classes) #nn.Linear(512 * self.block.expansion, len(SCENE_2_LABEL_2_ID_MAPPING[Scene.INDIAN_PINES]))
        self.remotesensing_salinas_fc = nn.Linear(512 * self.block.expansion, num_classes) #nn.Linear(512 * self.block.expansion, len(SCENE_2_LABEL_2_ID_MAPPING[Scene.SALINAS]))
        self.remotesensing_paviau_fc = nn.Linear(512 * self.block.expansion, num_classes) #nn.Linear(512 * self.block.expansion, len(SCENE_2_LABEL_2_ID_MAPPING[Scene.PAVIA_UNIVERSITY]))

        #self.fruit_ripeness_fc = nn.Linear(512 * self.block.expansion, num_classes) #nn.Linear(512 * self.block.expansion, 3)
        self.avocado_ripeness_fc = nn.Linear(512 * self.block.expansion, num_classes)
        self.kiwi_ripeness_fc = nn.Linear(512 * self.block.expansion, num_classes)
        self.mango_ripeness_fc = nn.Linear(512 * self.block.expansion, num_classes)
        self.kaki_ripeness_fc = nn.Linear(512 * self.block.expansion, num_classes)
        self.papaya_ripeness_fc = nn.Linear(512 * self.block.expansion, num_classes)
        #self.fruit_firmness_fc = nn.Linear(512 * self.block.expansion, num_classes) #nn.Linear(512 * self.block.expansion, 3)
        self.avocado_firmness_fc = nn.Linear(512 * self.block.expansion, num_classes)
        self.kiwi_firmness_fc = nn.Linear(512 * self.block.expansion, num_classes)
        self.mango_firmness_fc = nn.Linear(512 * self.block.expansion, num_classes)
        self.kaki_firmness_fc = nn.Linear(512 * self.block.expansion, num_classes)
        self.papaya_firmness_fc = nn.Linear(512 * self.block.expansion, num_classes)
        #self.fruit_sugar_fc = nn.Linear(512 * self.block.expansion, num_classes) #nn.Linear(512 * self.block.expansion, 3)
        self.kiwi_sugar_fc = nn.Linear(512 * self.block.expansion, num_classes)
        self.mango_sugar_fc = nn.Linear(512 * self.block.expansion, num_classes)
        self.kaki_sugar_fc = nn.Linear(512 * self.block.expansion, num_classes)
        self.papaya_sugar_fc = nn.Linear(512 * self.block.expansion, num_classes)

        self.debris_patchwise_fc = nn.Linear(512 * self.block.expansion, num_classes) #nn.Linear(512 * self.block.expansion, len(CLASS_LABEL_2_ID_MAPPING))
        self.debris_objectwise_fc = nn.Linear(512 * self.block.expansion, num_classes) #nn.Linear(512 * self.block.expansion, len(CLASS_LABEL_2_ID_MAPPING))

        # self.group_configs = ['remote_sensing/indian_pines/.*', 'remote_sensing/salinas/.*', 'remote_sensing/paviaU/.*',
        #                       'fruit/.*/ripeness/.*', 'fruit/.*/firmness/.*', 'fruit/.*/sugar/.*',
        #                       'debris/.*/patchwise', 'debris/.*/objectwise']
        self.group_configs = ['remote_sensing/indian_pines/.*', 'remote_sensing/salinas/.*', 'remote_sensing/paviaU/.*',
                              'fruit/avocado/ripeness/.*', 'fruit/kiwi/ripeness/.*', 'fruit/mango/ripeness/.*', 'fruit/kaki/ripeness/.*', 'fruit/papaya/ripeness/.*',
                              'fruit/avocado/firmness/.*', 'fruit/kiwi/firmness/.*', 'fruit/mango/firmness/.*', 'fruit/kaki/firmness/.*', 'fruit/papaya/firmness/.*',
                              'fruit/kiwi/sugar/.*', 'fruit/mango/sugar/.*', 'fruit/kaki/sugar/.*', 'fruit/papaya/sugar/.*',
                              'debris/.*/patchwise', 'debris/.*/objectwise']
        # self.group_heads = [self.remotesensing_indianpines_fc, self.remotesensing_salinas_fc, self.remotesensing_paviau_fc,
        #             self.fruit_ripeness_fc, self.fruit_firmness_fc, self.fruit_sugar_fc,
        #             self.debris_patchwise_fc, self.debris_objectwise_fc]
        self.group_heads = [self.remotesensing_indianpines_fc, self.remotesensing_salinas_fc, self.remotesensing_paviau_fc,
                            self.avocado_ripeness_fc, self.kiwi_ripeness_fc, self.mango_ripeness_fc, self.kaki_ripeness_fc, self.papaya_ripeness_fc,
                            self.avocado_firmness_fc, self.kiwi_firmness_fc, self.mango_firmness_fc, self.kaki_firmness_fc, self.papaya_firmness_fc,
                            self.kiwi_sugar_fc, self.mango_sugar_fc, self.kaki_sugar_fc, self.papaya_sugar_fc,
                            self.debris_patchwise_fc, self.debris_objectwise_fc]

        self.init_params()

    def forward(self, x, meta_data=None):
        assert meta_data is not None
        configs = np.array([m.config for m in meta_data])
        #print(configs, '\n')

        channel_wavelengths = get_channel_wavelengths(meta_data).type_as(x)

        x = self.conv1(x, channel_wavelengths=channel_wavelengths)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        out = []
        for c, fc in zip(self.group_configs, self.group_heads):
            o = x[[re.search(c, ci) is not None for ci in configs]]
            if o.shape[0] > 0:
                out.append(fc(o))
        out = torch.concatenate(out)

        return out

    def reset_head(self, seed=0, num_classes=None, BN=False):
        print('Reset head')
        if num_classes is None:
            #self.init_params(layers=self.fc, Norm=False, seed=seed)
            for h in self.group_heads:
                self.init_params(layers=h, Norm=BN, seed=seed)
        else:
            print('-> {} classes'.format(num_classes))
            self.num_classes = num_classes

            # #self.fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            #
            # self.remotesensing_indianpines_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            # self.remotesensing_salinas_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            # self.remotesensing_paviau_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            #
            # #self.fruit_ripeness_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            # self.avocado_ripeness_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            # self.kiwi_ripeness_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            # self.mango_ripeness_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            # self.kaki_ripeness_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            # self.papaya_ripeness_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            # #self.fruit_firmness_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            # self.avocado_firmness_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            # self.kiwi_firmness_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            # self.mango_firmness_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            # self.kaki_firmness_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            # self.papaya_firmness_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            # #self.fruit_sugar_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            # self.kiwi_sugar_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            # self.mango_sugar_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            # self.kaki_sugar_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            # self.papaya_sugar_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            #
            # self.debris_patchwise_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            # self.debris_objectwise_fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            #
            # # self.group_heads = [self.remotesensing_indianpines_fc, self.remotesensing_salinas_fc, self.remotesensing_paviau_fc,
            # #                     self.fruit_ripeness_fc, self.fruit_firmness_fc, self.fruit_sugar_fc,
            # #                     self.debris_patchwise_fc, self.debris_objectwise_fc]
            # self.group_heads = [self.remotesensing_indianpines_fc, self.remotesensing_salinas_fc, self.remotesensing_paviau_fc,
            #                     self.avocado_ripeness_fc, self.kiwi_ripeness_fc, self.mango_ripeness_fc, self.kaki_ripeness_fc, self.papaya_ripeness_fc,
            #                     self.avocado_firmness_fc, self.kiwi_firmness_fc, self.mango_firmness_fc, self.kaki_firmness_fc, self.papaya_firmness_fc,
            #                     self.kiwi_sugar_fc, self.mango_sugar_fc, self.kaki_sugar_fc, self.papaya_sugar_fc,
            #                     self.debris_patchwise_fc, self.debris_objectwise_fc]
            #
            # #self.init_params(layers=self.fc, Norm=BN, seed=seed)
            # for h in self.group_heads:
            #     self.init_params(layers=h, Norm=BN, seed=seed)

            fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            # for n, p in fc.named_parameters():
            #     print(n, p)

            self.init_params(layers=nn.Sequential(fc), Norm=BN, seed=seed)
            # for n, p in fc.named_parameters():
            #     print(n, p)

            self.remotesensing_indianpines_fc, self.remotesensing_salinas_fc, self.remotesensing_paviau_fc, self.avocado_ripeness_fc, self.kiwi_ripeness_fc, self.mango_ripeness_fc, self.kaki_ripeness_fc, self.papaya_ripeness_fc, self.avocado_firmness_fc, self.kiwi_firmness_fc, self.mango_firmness_fc, self.kaki_firmness_fc, self.papaya_firmness_fc, self.kiwi_sugar_fc, self.mango_sugar_fc, self.kaki_sugar_fc, self.papaya_sugar_fc, self.debris_patchwise_fc, self.debris_objectwise_fc = (3 + 5+ 5+ 4 +2) * [copy.deepcopy(fc)]

            self.group_heads = [self.remotesensing_indianpines_fc, self.remotesensing_salinas_fc,
                                self.remotesensing_paviau_fc,
                                self.avocado_ripeness_fc, self.kiwi_ripeness_fc, self.mango_ripeness_fc,
                                self.kaki_ripeness_fc, self.papaya_ripeness_fc,
                                self.avocado_firmness_fc, self.kiwi_firmness_fc, self.mango_firmness_fc,
                                self.kaki_firmness_fc, self.papaya_firmness_fc,
                                self.kiwi_sugar_fc, self.mango_sugar_fc, self.kaki_sugar_fc, self.papaya_sugar_fc,
                                self.debris_patchwise_fc, self.debris_objectwise_fc]


            # for n, p in self.group_heads[0].named_parameters():
            #     print(n, p)
            # for n, p in self.remotesensing_indianpines_fc.named_parameters():
            #     print(n, p)
            #
            # for n, p in self.group_heads[5].named_parameters():
            #     print(n, p)



if __name__ == '__main__':
    #mt_model = Multihead_DeepHSNet_with_HyVEConv(wavelength_range=(400, 1000), num_of_wrois=5, num_classes=3)
    mt_model = Multihead_ResNet_with_HyVEConv(wavelength_range=(400, 1000), num_of_wrois=5, num_classes=3)
    mt_model.reset_head(num_classes=4, BN=False, seed=5)
    mt_model.reset(seed=6)
