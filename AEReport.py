#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from collections import Counter
import numpy as np

# This is a Report of Auto Encoder Expriment with matplotlib
class AEReport(object):
    # To define the position of the graphs and images.
    repdef = {
                # (0.Row Start Position, 1.Colmn Start Position, 2.Row Count, 3.Colmn Count)
                 'Header': (0, 0, 3, 29),      # 0.Header
                 'Footer': (32, 0, 3, 14),     # 1.Footer
                 'W': (4, 3, 17, 10),           # 2.W (Hidden Layer)
                 'W_range': (4, 2, 1, 1),      # 3.W_range (Min/Mean/Max)
                 'b_range': (4, 13, 1, 1),     # 4.b_range
                 'x_hat': (4, 17, 17, 12),     # 5.x_hat (encode/decode)
                 'MNIST': (21, 3, 1, 10),      # 6.MNIST ORINAL DATA
                 'TRAIN': (21, 17, 1, 10),     # 7.Training DATA
                 'Calibration': (21, 27, 1, 2),  # 8.Calibration Data
                 'Label1': (3, 0, 1, 2),      # 9.Label  period/ ...
                 'Label2': (3, 2, 1, 1),      # 10.Label  W range ...
                 'Label3': (3, 3, 1, 10),     # 11.Label  W (Hidden...
                 'Label4': (3, 13, 1, 1),     # 12.Label  b range ...
                 'Label5': (3, 14, 1, 2),     # 13.Label  Time...
                 'Label6': (3, 17, 1, 12),    # 14.Label  x_hat of...
                 'Label7': (3, 16, 1, 1),     # 15.Label  cost ...
                 'Index1': (4, 0, 16, 2),     # 16.Index Period / ***
                 'Index2': (4, 14, 16, 2),    # 17.Index  bmax ***
                 'Index3': (21, 0, 1, 3),     # 18.Label   x (input)
                 'Index4': (21, 15, 1, 2),    # 19.Label  y
                 'Index5': (4, 16, 16, 1),    # 20.Index  bmax ***
                 'CostFig': (22, 2, 7, 9),    # 21.Cost Fig.
                 'LastWRangeFig': (22, 15, 4, 11),   # 22.Etc. Fig.
                 'LastZFig': (29, 15, 3, 11),  # 23.Etc. Fig.
                 'Etc.Fig': (22, 14, 9, 11),   # 24.Etc. Fig.
    }

    def __init__(self):
        self.fig = plt.figure(figsize=(14, 19))
        self.gs = gridspec.GridSpec(34, 29)
        self.gs.update(wspace=0.0, hspace=0.1)
        self.var_offset = 0
        self.last_ent_diff = 0.0
        self.last_img_diff = 0
        self.initfilter = []
        plt.style.use('ggplot')

    def draw_digit(self, data, pos_r, pos_c, n_size):
        ax1 = plt.subplot(self.gs[pos_r, pos_c])
        plt.gray()
        #            for sp in ax1.spines.values():
        #                sp.set_visible(False)
        ax1.set_xticks([])
        ax1.set_yticks([])
        Z = data.reshape(n_size, n_size)
        Z = Z[::-1, :]
        ax1.pcolor(Z)

    def draw_digit_a(self, data, pos_r, pos_c, n_size):
        ax1 = plt.subplot(self.gs[pos_r, pos_c])
        plt.gray()
        #            for sp in ax1.spines.values():
        #                sp.set_visible(False)
        ax1.set_xticks([])
        ax1.set_yticks([])
        Z = data.reshape(n_size, n_size)
        Z = Z[::-1, :]
        ax1.pcolor(Z, vmin=0.0, vmax=0.3) #


    def draw_cost(self, my_ae, my_train):
        data = my_ae.get_costrec()
        rd = self.repdef['CostFig']
        ax2 = plt.subplot(self.gs[rd[0]:rd[0] + rd[2] + 1, rd[1]:(rd[1] + rd[3])])
        ax2tw = ax2.twinx()
        ax2.plot(data, lw=1)
        # ax2.set_title("Cost/Epoch",  horizontalalignment='left', verticalalignment='bottom')
        ax2.set_ylabel("Cost  (log scale)")
        ax2.set_xlabel("Cost Fig.  horizontal axis=epoch")
        ax2.set_yscale('log')
        #ax2.set_ylim(plt.ylim()[0], 10.0) #Upper Limit 10.0
        #ax2.set_ylim(1, 10)

        print(" Fig. Cost/epoch")
        ent_digit = [[], [], [], [], [], [], [], [], [], [], ]
        for i in range(np.int(np.log2(my_ae.epoch_limit) + 1)):
            ent = self.delta_x_entropy(my_ae, my_train, i)
            #print("period:%d " % i)
            #print(ent)
            for j in range(10):
                ent_digit[j].append(ent[j])
        x = 2 ** np.array(range(np.int(np.log2(my_ae.epoch_limit) + 1)))
        #print(x)
        #print(ent_digit[0])
        for i in range(10):
            ax2tw.plot(x, np.array(ent_digit[i]) / 100., label=str(i))
            ax2tw.legend(bbox_to_anchor=(0.85, 1), loc='upper left', borderaxespad=0, prop={'size': 8})
        ax2tw.set_ylabel("total entropy(|y - x_hat|)/100")

    def draw_last_Wb_range(self, my_ae, my_train):
        my_period = np.int(np.log2(my_ae.epoch_limit) + 1)
        dataW = my_ae.get_W1(my_period)
        dataB = my_ae.get_b1(my_period)
        rd = self.repdef['LastWRangeFig']
        ax3 = plt.subplot(self.gs[rd[0]:rd[0] + rd[2] + 1, rd[1]:(rd[1] + rd[3])])
        ax3tw = ax3.twinx()
        x = range(len(dataW))
        y_mean = np.array([])
        y_max = np.array([])
        y_min = np.array([])

        for i in x:
            y_mean = np.append(y_mean, np.array(np.mean([dataW[i]])))
            y_max = np.append(y_max, np.array(np.max([dataW[i]])))
            y_min = np.append(y_min, np.array(np.min([dataW[i]])))
        y_low = y_mean - y_min
        y_up = y_max - y_mean
        a_err = [y_low, y_up]
        ax3.errorbar(x, y_mean, yerr=a_err, label="W")
        ax3tw.plot(x, dataB,  'd', markersize=4, markerfacecolor='blue', label="b")
        # ax2.set_title("Cost/Epoch",  horizontalalignment='left', verticalalignment='bottom')
        ax3.set_ylabel("W Min/Mean/max")
        ax3.set_xlabel("Last W Range & b Bias Fig.  horiz axis=Node Number")
        ax3.set_xlim(-1, len(y_mean))
        #Reset Y Limit (Adjust Zero Lebel)
        #w_limit = math.ceil(np.max((np.max(dataW), -np.min(dataW))))
        w_limit = np.max((np.max(dataW), -np.min(dataW)))
        ax3.set_ylim(-w_limit, w_limit)
        #b_limit = math.ceil(np.max((np.max(dataB), -np.min(dataB))))
        b_limit = np.max((np.max(dataB), -np.min(dataB))) + 0.2
        ax3tw.set_ylim(-b_limit, b_limit)
        ax3tw.set_ylabel("b Bias plot")
        ax3.legend(bbox_to_anchor=(1.11, 0.0), loc='lower left', borderaxespad=0, prop={'size': 8})
        ax3tw.legend(bbox_to_anchor=(1.11, 1.0), loc='upper left', borderaxespad=0, prop={'size': 8})
        ax3.xaxis.set_minor_locator(tick.MultipleLocator(2))

        print(" Fig. W Min/Mean/Max")

    def draw_last_Z_range(self, my_ae, my_train):
        my_period = np.int(np.log2(my_ae.epoch_limit) + 1)
        dataW = my_ae.get_W1(my_period)
        dataB = my_ae.get_b1(my_period)
        rd = self.repdef['LastZFig']
        ax4 = plt.subplot(self.gs[rd[0]:rd[0] + rd[2] + 1, rd[1]:(rd[1] + rd[3])])
        x = range(len(dataW))
        zshape = my_ae.encode_by_snap(dataW, dataB, my_train[my_ae.get_mnist_start_index(0) + self.var_offset])
        zsum = np.zeros(zshape.shape)
        for sample in range(10):
            z = my_ae.encode_by_snap(dataW, dataB, my_train[my_ae.get_mnist_start_index(sample) + self.var_offset])
            zsum += z
            x = range(len(z))
            ax4.plot(x, z + (9.0 - sample), label=str(sample))
        ax4.plot(x, zsum, color="b", label="Sum")
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,  prop={'size' : 6})
        ax4.set_xlabel("Last Z=f(Wx+b) Range Fig.  horiz axis=Node Number")
        ax4.xaxis.set_minor_locator(tick.MultipleLocator(2))
        ax4.yaxis.set_minor_locator(tick.MultipleLocator(1))

    def report_header(self, my_ae):
        c1, c2, c3, c4, c5, c6 = 0.17, 0.34, 0.48, 0.62, 0.76, 0.87
        r1, r2, r3, r4, r5 = 0.7, 0.56, 0.42, 0.28, 0.14
        sl, sm, ss = 14, 12, 9
        f1 = 'serif'

        title = [("Auto Encoder Experiment Report", 0.01, 0.94, sl, f1, 'left', 'top'),
                 ("Experiment-ID:",     c6, 0.92, sm, f1, 'right', 'top'),
                 ("Condition:",     0.01, r1, sm, f1, 'left', 'top'),
                 ("Node Count:",    c1, r1, ss, f1, 'right', 'top'),
                 ("Activation Func:",   c1, r2, ss, f1, 'right', 'top'),
                 ("BatchSize:",       c1, r3, ss, f1, 'right', 'top'),
                 ("Noise Ratio:",      c1, r4, ss, f1, 'right', 'top'),
                 ("Epoch Limit:",     c2, r1, ss, f1, 'right', 'top'),
                 ("Drop Out:",   c2, r2, ss, f1, 'right', 'top'),
                 ("Train Shuffle:",     c2, r3, ss, f1, 'right', 'top'),
                 ("W untied:",   c2, r4, ss, f1, 'right', 'top'),
                 ("Alpha Ratio:",    c3, r1, ss, f1, 'right', 'top'),
                 ("A*Bias Ratio:",     c3, r2, ss, f1, 'right', 'top'),
                 ("Beta Ratio:",      c3, r3, ss, f1, 'right', 'top'),
                 (":",    c3, r4, ss, f1, 'right', 'top'),
                 ("Train SubScale:",    c4, r1, ss, f1, 'right', 'top'),
                 ("W Transport:",     c4, r2, ss, f1, 'right', 'top'),
                 ("Normalization:",      c4, r3, ss, f1, 'right', 'top'),
                 ("Whitening:",    c4, r4, ss, f1, 'right', 'top'),
                 ("Optimizer:",    c5, r1, ss, f1, 'right', 'top'),
                 ("6:",     c5, r2, ss, f1, 'right', 'top'),
                 ("Option7:",      c5, r3, ss, f1, 'right', 'top'),
                 ("Option8:",    c5, r4, ss, f1, 'right', 'top'),
                 ("Researcher:",     c6, r1, ss, f1, 'right', 'top'),
                 ("DateTime:",      c6, r2, ss, f1, 'right', 'top'),
                 ("FrameWork:",     c6, r3, ss, f1, 'right', 'top'),
                 ("CPU/GPU:",       c6, r4, ss, f1, 'right', 'top'),
                 ("Note:",     0.01, r5, sm, f1, 'left', 'top'),
                 ]
        line = [(0.97, 2), (0.74, 2), (0.16, 1), (0.01, 1)]
        data = [
            (my_ae.exp_id, c6, 0.92, sm, f1, 'left', 'top'),
            (my_ae.n_hidden, c1, r1, ss, f1, 'left', 'top'),
            (my_ae.func.name, c1, r2, ss, f1, 'left', 'top'),
            (my_ae.batch_size, c1, r3, ss, f1, 'left', 'top'),
            (my_ae.noise, c1, r4, ss, f1, 'left', 'top'),
            (my_ae.epoch_limit, c2, r1, ss, f1, 'left', 'top'),
            (my_ae.dropout, c2, r2, ss, f1, 'left', 'top'),
            (my_ae.shuffle, c2, r3, ss, f1, 'left', 'top'),
            (my_ae.untied, c2, r4, ss, f1, 'left', 'top'),
            (my_ae.alpha, c3, r1, ss, f1, 'left', 'top'),
            (my_ae.alphaBias, c3, r2, ss, f1, 'left', 'top'),
            (my_ae.beta, c3, r3, ss, f1, 'left', 'top'),
            ("", c3, r4, ss, f1, 'left', 'top'),
            ("1/" + str(my_ae.sub_scale), c4, r1, ss, f1, 'left', 'top'),
            (my_ae.Wtransport, c4, r2, ss, f1, 'left', 'top'),
            (my_ae.normalization, c4, r3, ss, f1, 'left', 'top'),
            (my_ae.whitening, c4, r4, ss, f1, 'left', 'top'),
            (my_ae.optimizer, c5, r1, ss, f1, 'left', 'top'),
            ("", c5, r2, ss, f1, 'left', 'top'),
            ("", c5, r3, ss, f1, 'left', 'top'),
            ("", c5, r4, ss, f1, 'left', 'top'),

            ("Koo Wells", c6, r1, ss, f1, 'left', 'top'),
            (my_ae.exp_date, c6, r2, ss, f1, 'left', 'top'),
            ("None", c6, r3, ss, f1, 'left', 'top'),
            ("Core i7 2700K/None", c6, r4, ss, f1, 'left', 'top'),
            (my_ae.note, 0.05, 0.01, ss, f1, 'left', 'bottom'),

        ]
        rd = self.repdef['Header']
        axhead = plt.subplot(self.gs[rd[0]:(rd[0] + rd[2]), rd[1]:(rd[1] + rd[3])])
        axhead.set_xticks([])
        axhead.set_yticks([])
        for i in range(len(title)):
            axhead.annotate(title[i][0], xy=(title[i][1], title[i][2]), xycoords='axes fraction',
                            fontsize=title[i][3],
                            fontname=title[i][4],
                            horizontalalignment=title[i][5], verticalalignment=title[i][6],
                            )
        for i in range(len(line)):
            self.report_line(axhead, line[i][0], line[i][1])
        for i in range(len(data)):
            axhead.annotate(str(data[i][0]), xy=(data[i][1] + 0.005, data[i][2]), xycoords='axes fraction',
                            fontsize=data[i][3],
                            fontname=data[i][4],
                            horizontalalignment=data[i][5], verticalalignment=data[i][6],
                            )

        print(" Header")

    def report_line(self, ax, my_y, my_w):
        ax.axhline(y=my_y, xmin=0., xmax=1.0, linewidth=my_w, color='k')

    def report_footer(self, my_ae, my_train):
        c1, c2, c3, c4, c5, c6 = 0.4, 0.7, 0.0, 0.0, 0.0, 0.0
        r1, r2, r3, r4, r5 = 0.97, 0.73, 0.49, 0.25, 0.0
        sl, sm, ss = 11, 10, 9
        f1 = 'serif'
        use_W = 0.0
        for sample in range(10):
            z = my_ae.encode_by_snap(my_ae.get_W1(np.int(np.log2(my_ae.epoch_limit)) + 1), my_ae.get_b1(np.int(np.log2(my_ae.epoch_limit)) + 1), my_train[my_ae.get_mnist_start_index(sample)])
            use_W += float(np.sum(1 * (np.abs(z) > 0.2))) / len(z)
        userW_ratio = use_W / 10
        wreg, wvar = self.region_ratio(my_ae)
        norm_L1 = 0.0
        norm_L2 = 0.0
        mean_sp = 0.0
        for sample in range(10):
            z = my_ae.encode_by_snap(my_ae.get_W1(np.int(np.log2(my_ae.epoch_limit)) + 1),
                                     my_ae.get_b1(np.int(np.log2(my_ae.epoch_limit)) + 1),
                                     my_train[my_ae.get_mnist_start_index(sample)])
            mean_sp += np.sum((np.abs(z) / np.max(z)) ** 0.5)
        sp_simple_ratio = mean_sp / len(z) / 10 * 100

        footer = [
                  ("Total Training time(sec):", c1, r1, sl, f1, 'right', 'top'),
                  ("{0:.1f}".format(my_ae.traintime), c1, r1, sl, f1, 'left', 'top'),
                  ("W EntropyGain:", c1, r2, sl, f1, 'right', 'top'),
                  ("{0:.1f}".format(self.delta_entropy(my_ae.get_W1(0), my_ae.get_W1(np.int(np.log2(my_ae.epoch_limit)) + 1))), c1, r2, sl, f1, 'left', 'top'),
                  ("Sparse simple Ratio(%):", c1, r3, sl, f1, 'right', 'top'),
                  ("{0:.2f}".format(sp_simple_ratio), c1, r3, sl, f1, 'left', 'top'),
                  ("Last Entropy Diff:", c1, r4, sl, f1, 'right', 'top'),
                  ("{0:.1f}".format(self.last_ent_diff), c1, r4, sl, f1, 'left', 'top'),
                  ("W Region:", c2, r1, sl, f1, 'right', 'top'),
                  (wreg, c2, r1, sl, f1, 'left', 'top'),
                  ("W Var:", c2, r2, sl, f1, 'right', 'top'),
                  ("{0:.3f}".format(wvar), c2, r2, sl, f1, 'left', 'top'),
                  ("Img Diff:", c2, r4, sl, f1, 'right', 'top'),
                  ("{0:.1f}".format(self.last_img_diff), c2, r4, sl, f1, 'left', 'top'),
                ]
        rd = self.repdef['Footer']
        axfoot = plt.subplot(self.gs[rd[0]:(rd[0] + rd[2]), rd[1]:(rd[1] + rd[3])])
        axfoot.set_xticks([])
        axfoot.set_yticks([])
        for i in range(len(footer)):
            axfoot.annotate(footer[i][0], xy=(footer[i][1], footer[i][2]), xycoords='axes fraction',
                            fontsize=footer[i][3],
                            fontname=footer[i][4],
                            horizontalalignment=footer[i][5], verticalalignment=footer[i][6],
                            )
        print(" Footer")

    def region_ratio(self, my_ae):
        period = np.int(np.log2(my_ae.epoch_limit) + 1)
        r = 0
        v = 0
        for j in range(len(my_ae.get_W1(period))):
            a = np.abs(my_ae.get_W1(period)[j] - np.mean(my_ae.get_W1(period)[j]) *
                       np.ones(my_ae.get_W1(period)[j].shape))
            b = np.mean(a) * np.ones(my_ae.get_W1(period)[j].shape)
            v += np.sum(np.var(a))
            for i in range(100):
                r += np.sum(1 * (a > b * (i + 3)))
        r = 1.0 * r / my_ae.n_hidden
        return r, v

    def report_label(self, my_ae, my_select):
        c1, c2, c3, c4, c5, c6 = 0.01, 0.05, 0.04, 0.12, 0.95, 0.99
        r1, r2, r3, r4, r5, r6 = 0.98, 0.9, 0.64, 0.05, 0.31, 0.01
        sl, sm, ss, sll = 11, 8, 7, 13
        f1 = 'serif'
        index = [[("period/", c1, r2, sm, f1, 'left', 'top'),
                 ("(epoch)", 0, r4, sm, f1, 'left', 'bottom'),
                 ("Wmax", c6, r1, ss, f1, 'right', 'top'),
                 ("mean", c6, r3, ss, f1, 'right', 'top'),
                 ("Wmin", c6, r5, ss, f1, 'right', 'top'),
                  ],
                 [("W", c2, r2, sll, f1, 'left', 'top'),
                  ("range", c5, r4, sm, f1, 'right', 'bottom'),
                  ],
                 [("W", c2, r2, sll, f1, 'left', 'top'),
                  ("(Hidden layer Weight)", c4, r2, sm, f1, 'left', 'top'),
                  ("10 Node of select", c5, r2, sm, f1, 'right', 'top'),
                  ],
                 [("b", c2, r2, sll, f1, 'left', 'top'),
                  ("range", c5, r4, sm, f1, 'right', 'bottom'),
                  ],
                 [("Time", c6, r1, ss, f1, 'right', 'top'),
                  ("Wgain", c6, r3, ss, f1, 'right', 'top'),
                  ("b max", c1, r1, ss, f1, 'left', 'top'),
                  (" mean", c1, r3, ss, f1, 'left', 'top'),
                  ("b min", c1, r5, ss, f1, 'left', 'top'),
                  ],
                 [("x_hat", c1, r1, sll, f1, 'left', 'top'),
                  ("of Sample MNIST DATA & calibration (encode/decode by W-Layer)", c4, r1, sm, f1, 'left', 'top'),
                  ],
                 [("Cost", c1, r1, ss, f1, 'left', 'top'),
                  ("Img-diff", c1, r3, ss, f1, 'left', 'top'),
                  ("ent-diff", c1, r5, ss, f1, 'left', 'top'),
                  ],
                 ]
        select_index = [[("[n]", c2, r4, sm, f1, 'center', 'bottom'),
                         ],
                        ]
        sub_index = [[("n", c3, r6, sm, f1, 'center', 'bottom'),
                      ],
                     ]

        j = 0
        for rd in [self.repdef['Label1'], self.repdef['Label2'], self.repdef['Label3'], self.repdef['Label4'],
                   self.repdef['Label5'], self.repdef['Label6'], self.repdef['Label7']]:
            axindex = plt.subplot(self.gs[rd[0], rd[1]:(rd[1] + rd[3])])
            axindex.set_xticks([])
            axindex.set_yticks([])
            for i in range(len(index[j])):
                axindex.annotate(index[j][i][0], xy=(index[j][i][1], index[j][i][2]), xycoords='axes fraction',
                                 fontsize=index[j][i][3],
                                 fontname=index[j][i][4],
                                 horizontalalignment=index[j][i][5], verticalalignment=index[j][i][6],
                                 )
            j += 1
        # Write the selected W number in Label3
        rd = self.repdef['Label3']
        axindex = plt.subplot(self.gs[rd[0], rd[1]:(rd[1] + rd[3])])
        axindex.set_xticks([])
        axindex.set_yticks([])
        j = 0
        for i in range(len(select_index)):
            for ii in range(len(my_select)):
                axindex.annotate("[" + str(my_select[ii]) + "]",
                                 xy=(select_index[j][i][1] + 0.1 * ii, select_index[j][i][2]),
                                 xycoords='axes fraction',
                                 fontsize=select_index[j][i][3], fontname=select_index[j][i][4],
                                 horizontalalignment=select_index[j][i][5], verticalalignment=select_index[j][i][6],
                                 )

    # Write Sub heading to Label6
        rd = self.repdef['Label6']
        axindex = plt.subplot(self.gs[rd[0], rd[1]:(rd[1] + rd[3])])
        axindex.set_xticks([])
        axindex.set_yticks([])
        j = 0
        for i in range(len(select_index)):
            sub6_index = ""
            for ii in range(12):
                if ii < 10:
                    sub6_index = str(ii)
                if ii == 10:
                    sub6_index = "Zero"
                if ii == 11:
                    sub6_index = "Mean"
                axindex.annotate(sub6_index,
                                 xy=(sub_index[j][i][1] + (1.0 / 12) * ii, sub_index[j][i][2]),
                                 xycoords='axes fraction',
                                 fontsize=sub_index[j][i][3], fontname=sub_index[j][i][4],
                                 horizontalalignment=sub_index[j][i][5], verticalalignment=sub_index[j][i][6],
                                 )

    def report_index1(self, my_ae):
        period = 0
        c1, c2, c3, c4, c5, c6 = 0.01, 0.05, 0.04, 0.0, 0.95, 0.99
        r1, r2, r3, r4, r5, r6 = 0.98, 0.9, 0.64, 0.05, 0.31, 0.01
        sl, sm, ss = 9, 8, 7.6
        f1 = 'serif'
        index = [("Period /", c2, r2, sl, f1, 'left', 'top'),     #
                 ("(Epoch)", 0, r4, sm, f1, 'left', 'bottom'),
                 ("Wmax", c6, r1, sm, f1, 'right', 'top'),
                 ("mean", c6, r3, sm, f1, 'right', 'top'),
                 ("Wmin", c6, r5, sm, f1, 'right', 'top'),
                 ]
        for period in range(np.int(np.log2(my_ae.epoch_limit)) + 2):  # Since output a initial state, range + 2
            rd = self.repdef['Index1']
            axindex = plt.subplot(self.gs[rd[0] + period, rd[1]:(rd[1] + rd[3])])
            axindex.set_xticks([])
            axindex.set_yticks([])
            for j in range(5):
                disp = ""
                if j == 0:
                    disp = str(period) + " /"
                if j == 1:
                    disp = "(" + str(int(2 ** (period - 1))) + ")"
                if j == 2:
                    disp = "{0:.3f}".format(np.max(my_ae.get_W1(period)))
                if j == 3:
                    disp = "{0:.3f}".format(np.mean(my_ae.get_W1(period)))
                if j == 4:
                    disp = "{0:.3f}".format(np.min(my_ae.get_W1(period)))

                axindex.annotate(disp, xy=(index[j][1], index[j][2]), xycoords='axes fraction',
                                 fontsize=index[j][3],
                                 fontname=index[j][4],
                                 horizontalalignment=index[j][5], verticalalignment=index[j][6],
                                 )

        print(" Index1")

    def report_index2(self, my_ae, my_train):
        c1, c2, c3, c4, c5, c6 = 0.01, 0.05, 0.04, 0.0, 0.95, 0.99
        r1, r2, r3, r4, r5, r6 = 0.98, 0.9, 0.64, 0.05, 0.31, 0.01
        sl, sm, ss = 9, 8, 7.6
        f1 = 'serif'
        index = [("bmax", c1, r1, sm, f1, 'left', 'top'),
                 ("mean", c1, r3, sm, f1, 'left', 'top'),
                 ("bmin", c1, r5, sm, f1, 'left', 'top'),
                 ("Time", c6, r1, sm, f1, 'right', 'top'),
                 ("Wgain", c6, r3, sm, f1, 'right', 'top'),
                 ]
        for period in range(np.int(np.log2(my_ae.epoch_limit)) + 2):    # Since output a initial state, range + 2
            rd = self.repdef['Index2']
            axindex = plt.subplot(self.gs[rd[0] + period, rd[1]:(rd[1] + rd[3])])
            axindex.set_xticks([])
            axindex.set_yticks([])
            for j in range(5):
                disp = ""

                if j == 0:
                    disp = "{0:.3f}".format(np.max(my_ae.get_b1(period)))
                if j == 1:
                    disp = "{0:.3f}".format(np.mean(my_ae.get_b1(period)))
                if j == 2:
                    disp = "{0:.3f}".format(np.min(my_ae.get_b1(period)))
                if j == 3:
                    try:
                        if period == 0:
                            disp = "0.00"
                        else:
                            disp = "{0:.1f}".format(my_ae.extime[period - 1])
                    except:
                        disp = "Error"
                if j == 4:
                    disp = "{0:.2f}".format(self.delta_entropy(my_ae.get_W1(0), my_ae.get_W1(period)))

                axindex.annotate(disp, xy=(index[j][1], index[j][2]), xycoords='axes fraction',
                                 fontsize=index[j][3],
                                 fontname=index[j][4],
                                 horizontalalignment=index[j][5], verticalalignment=index[j][6],
                                 )
        print(" Index2")

    def report_index5(self, my_ae, my_train):
        c1, c2, c3, c4, c5, c6 = 0.01, 0.05, 0.04, 0.0, 0.95, 0.99
        r1, r2, r3, r4, r5, r6 = 0.98, 0.9, 0.64, 0.05, 0.31, 0.01
        sl, sm, ss = 9, 8, 7.6
        f1 = 'serif'
        index = [("cost", c6, r1, sm, f1, 'right', 'top'),
                 ("imgdiff", c6, r3, sm, f1, 'right', 'top'),
                 ("entdiff", c6, r5, sm, f1, 'right', 'top'),
                 ]
        for period in range(np.int(np.log2(my_ae.epoch_limit)) + 2):    # Since output a initial state, range + 2
            rd = self.repdef['Index5']
            axindex = plt.subplot(self.gs[rd[0] + period, rd[1]:(rd[1] + rd[3])])
            axindex.set_xticks([])
            axindex.set_yticks([])
            for j in range(3):
                disp = ""

                if j == 0:
                    disp = "{0:.3f}".format(my_ae.get_costrec()[int(2 ** (period - 1))])
                if j == 1:
                    digit_mean_ent, digit_x_ent,  digit_img_dif, digit_x_dif= self.get_ent_diff(my_ae, my_train, period)
                    self.last_ent_diff = digit_mean_ent
                    self.last_img_diff = digit_img_dif
                    disp = disp2 = "{0:.1f}".format(self.last_img_diff)
                if j == 2:
                    disp = "{0:.1f}".format(self.last_ent_diff)
                    print ("Entropy/Img Diff(" + str(period) + ") = " + disp + "/" + disp2)
                axindex.annotate(disp, xy=(index[j][1], index[j][2]), xycoords='axes fraction',
                                 fontsize=index[j][3],
                                 fontname=index[j][4],
                                 horizontalalignment=index[j][5], verticalalignment=index[j][6],
                                 )
        print(" index5")

    def get_ent_diff(self, my_ae, my_train, my_period):
        count = 100
        digit_x_ent = np.zeros((10, my_ae.n_visible))
        digit_x_dif = np.zeros((10, my_ae.n_visible))
        digit_mean_ent = 0.0
        digit_img_dif = 0.0
        # period = np.int(np.log2(my_ae.epoch_limit))
        for offset in range(count):
            for sample in range(10):
                org_img = my_train[my_ae.get_mnist_start_index(sample) + offset]
                y_x = my_ae.encode_by_snap(my_ae.get_W1(my_period), my_ae.get_b1(my_period), org_img)
                x_hat_img = my_ae.decode_by_snap(my_ae.get_W2(my_period), my_ae.get_b2(my_period), y_x)
                diff_img = np.abs(np.array(self.gray_scale(org_img)) - np.array(self.gray_scale(x_hat_img)))
                digit_x_dif[sample] += diff_img * diff_img
                # org_ent = self.entropy(self.gray_scale(org_img))
                # tgt_ent = self.entropy(self.gray_scale(x_hat_img))
                digit_x_ent[sample] += self.entropy(diff_img)
        for sample in range(10):
            digit_mean_ent += np.sum(digit_x_ent[sample]) / (count * 10)
            digit_img_dif += np.sum(digit_x_dif[sample]) ** 0.5 / count
        return digit_mean_ent, digit_x_ent, digit_img_dif, digit_x_dif

    def report_index3(self, my_ae):
        c1, c2, c3, c4, c5, c6 = 0.01, 0.05, 0.04, 0.0, 0.95, 0.99
        r1, r2, r3, r4, r5, r6 = 0.98, 0.9, 0.64, 0.05, 0.31, 0.01
        sl, sm, ss = 13, 8, 7
        f1 = 'serif'
        index3 = [("x", c2, r1, sl, f1, 'left', 'top'),
                  ("(input)", c6, r2, sm, f1, 'right', 'top'),
                  ("a Sample MNIST", c2, r4, sm, f1, 'left', 'bottom'),
                  ]
        rd = self.repdef['Index3']
        axindex3 = plt.subplot(self.gs[rd[0], rd[1]:(rd[1] + rd[3])])
        axindex3.set_xticks([])
        axindex3.set_yticks([])
        for i in range(len(index3)):
            axindex3.annotate(index3[i][0], xy=(index3[i][1], index3[i][2]), xycoords='axes fraction',
                              fontsize=index3[i][3],
                              fontname=index3[i][4],
                              horizontalalignment=index3[i][5], verticalalignment=index3[i][6],
                              )
        print(" index3")

    def report_index4(self, my_ae):
        c1, c2, c3, c4, c5, c6 = 0.01, 0.05, 0.04, 0.0, 0.95, 0.99
        r1, r2, r3, r4, r5, r6 = 0.98, 0.9, 0.64, 0.05, 0.31, 0.01
        sl, sm, ss = 13, 8, 7
        f1 = 'serif'
        index4 = [("y", c2, r1, sl, f1, 'left', 'top'),
                  ("training data", c6, r4, sm, f1, 'right', 'bottom'),
                  ]
        rd = self.repdef['Index4']
        axindex4 = plt.subplot(self.gs[rd[0], rd[1]:(rd[1] + rd[3])])
        axindex4.set_xticks([])
        axindex4.set_yticks([])
        for i in range(len(index4)):
            axindex4.annotate(index4[i][0], xy=(index4[i][1], index4[i][2]), xycoords='axes fraction',
                              fontsize=index4[i][3],
                              fontname=index4[i][4],
                              horizontalalignment=index4[i][5], verticalalignment=index4[i][6],
                              )
        print(" index4")

    def report_w_x_hat(self, my_ae, my_train, my_select):
        for period in range(np.int(np.log2(my_ae.epoch_limit)) + 2):    # Since output a initial state, range + 2
            targetW = my_ae.get_W1(period)
            rd = self.repdef['W']
            rd2 = self.repdef['x_hat']
            for sample in range(10):
                # Hidden Layer W
                self.draw_digit(targetW[my_select[sample]], rd[0] + period, rd[1] + sample, my_ae.n_size)

                # Encode/Decode Data
                y_x = my_ae.encode_by_snap(my_ae.get_W1(period), my_ae.get_b1(period),
                                           my_train[my_ae.get_mnist_start_index(sample) + self.var_offset])
                x_hat = my_ae.decode_by_snap(my_ae.get_W2(period), my_ae.get_b2(period), y_x)
                self.draw_digit(x_hat, rd2[0] + period, rd2[1] + sample, my_ae.n_size)
            # Calibration
            for i in range(len(self.m_cal)):
                y_x = my_ae.encode_by_snap(my_ae.get_W1(period), my_ae.get_b1(period),
                                           self.m_cal[i])
                x_hat = my_ae.decode_by_snap(my_ae.get_W2(period), my_ae.get_b2(period), y_x)
                self.draw_digit(x_hat, rd2[0] + period, rd2[1] + 10 + i, my_ae.n_size)

            print(" x_hat Fig. Period %d (Epoch %d)" % (period, 2 ** (period - 1)))

    def report_w_range(self, my_ae, my_select):
        last_period = np.int(np.log2(my_ae.epoch_limit))
        last_W = my_ae.get_W1(last_period)
        w_limit = np.max((np.max(last_W), -np.min(last_W)))
        for period in range(np.int(np.log2(my_ae.epoch_limit)) + 2):    # Since output a initial state, range + 2
            targetW = my_ae.get_W1(period)
            rd = self.repdef['W_range']
            # Hidden Layer W
            self.draw_one_w_range(targetW, my_select, rd[0] + period, rd[1], w_limit)

        print(" W Range Fig.")

    def report_b_range(self, my_ae, my_select):
        last_period = np.int(np.log2(my_ae.epoch_limit))
        last_B = my_ae.get_b1(last_period)
        b_limit = np.max((np.max(last_B), -np.min(last_B)))
        for period in range(np.int(np.log2(my_ae.epoch_limit)) + 2):    # Since output a initial state, range + 2
            targetB = my_ae.get_b1(period)
            rd = self.repdef['b_range']
            # Bias b
            self.draw_one_b_range(targetB, my_select, rd[0] + period, rd[1], b_limit)

        print(" Bias b Fig.")

    def draw_one_w_range(self, my_W, my_select, pos_r, pos_c, y_limit):
        y_mean = np.array([])
        y_max = np.array([])
        y_min = np.array([])
        x = range(10)
        for i in x:
            y_mean = np.append(y_mean, np.array(np.mean([my_W[my_select[i]]])))
            y_max = np.append(y_max, np.array(np.max([my_W[my_select[i]]])))
            y_min = np.append(y_min, np.array(np.min([my_W[my_select[i]]])))
        y_low = y_mean - y_min
        y_up = y_max - y_mean
        a_err = [y_low, y_up]
        ax = plt.subplot(self.gs[pos_r, pos_c])
        ax.errorbar(x, y_mean, yerr=a_err)
        ax.set_xlim(-1, 10)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_xticks([])
        ax.set_yticks([])

    def draw_one_b_range(self, my_b, my_select, pos_r, pos_c, y_limit):
        y_mean = np.zeros(10)
        y_max = np.array([])
        y_min = np.zeros(10)
        x = range(10)
        for i in x:
            y_max = np.append(y_max, np.array(np.mean([my_b[my_select[i]]])))
        y_low = y_mean - y_min
        y_up = y_max - y_mean
        a_err = [y_low, y_up]
        ax = plt.subplot(self.gs[pos_r, pos_c])
        ax.errorbar(x, y_mean, yerr=a_err, color="b")
        ax.set_xlim(-1, 10)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_xticks([])
        ax.set_yticks([])

    # Test method
    def report_w3d(self, my_ae):
        for period in range(np.int(np.log2(my_ae.epoch_limit) + 1)):
            targetW = my_ae.get_W1(period)
            x = range(27)
            y = range(27)

        ax3d = plt.subplot(self.gs[period, 2])
        X, Y = np.meshgrid(x, y)
        Z = targetW[2][X + (784 - 28) - Y * 28]
        fig = plt.figure()
        #ax = Axes3D(fig)
        ax = fig.gca(projection='3d')
        # ax.plot_wireframe(X,Y,Z)
        plt.cool()
        cset = ax.contourf(X, Y, Z, zdir='z', offset=-4, cmap=cm.coolwarm)
        cset = ax.contourf(X, Y, Z, zdir='x', offset=-1, cmap=cm.cool)
        cset = ax.contourf(X, Y, Z, zdir='y', offset=-1, cmap=cm.cool)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.3)
        # ax.contourf3D(X,Y,Z)
        # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        #                       linewidth=0, antialiased=False)
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.view_init(20, 30)
        ax.set_xlabel('X')
        ax.set_xlim(0, 27)
        ax.set_ylabel('Y')
        ax.set_ylim(0, 27)
        ax.set_zlabel('Z')
        ax.set_zlim(-4, 3)
        plt.show()

    def report_mnist(self, my_ae, my_train):
        # Original Minist Data Image
        rd = self.repdef['MNIST']
        for i in range(10):
            self.draw_digit(my_train[my_ae.get_mnist_start_index(i)], rd[0], rd[1] + i, my_ae.n_size)
        print(" Fig.  x  Original sample MNIST Data")

    def report_train(self, my_ae, my_train):
        # Original Minist Data Image
        rd = self.repdef['TRAIN']
        for i in range(10):
            self.draw_digit(my_train[my_ae.get_mnist_start_index(i)], rd[0], rd[1] + i, my_ae.n_size)
        print(" Fig.  y  Training Data")

    def report_calibration(self, my_ae, my_train):
        rd = self.repdef['Calibration']
        self.m_cal = []
        self.m_cal.append(np.zeros(my_train[0].shape))
        self.m_cal.append(np.mean(my_train) * np.ones(my_train[0].shape))
        for i in range(len(self.m_cal)):
            self.draw_digit_a(self.m_cal[i], rd[0], rd[1] + i, my_ae.n_size)
        print(" Calibration Data")

    def delta_entropy(self, org_W, target_W):
        sum_ent = 0.0
        for i in range(len(org_W)):
            org_ent = self.entropy(self.gray_scale(org_W[i]))
            tgt_ent = self.entropy(self.gray_scale(target_W[i]))
            sum_ent += (org_ent - tgt_ent)
        return sum_ent

    def delta_x_entropy(self, my_ae, my_train, my_period):
        count = 100
        digit_x_ent = np.zeros((10, 784))
        digit_mean_ent = np.zeros(10)
        #period = np.int(np.log2(my_ae.epoch_limit))
        for offset in range(count):
            for sample in range(10):
                org_img = my_train[my_ae.get_mnist_start_index(sample) + offset]
                y_x = my_ae.encode_by_snap(my_ae.get_W1(my_period), my_ae.get_b1(my_period), org_img)
                x_hat_img = my_ae.decode_by_snap(my_ae.get_W2(my_period), my_ae.get_b2(my_period), y_x)
                org_ent = self.entropy(self.gray_scale(org_img))
                tgt_ent = self.entropy(self.gray_scale(x_hat_img))
                digit_x_ent[sample] += np.abs(tgt_ent - org_ent)
        for sample in range(10):
            digit_mean_ent[sample] = np.sum(digit_x_ent[sample]) / count
        return digit_mean_ent

    def gray_scale(self, w):
        b = w - np.min(w) * np.ones(len(w))
        bb =  b / np.max(b + 0.00001) * 255.0
        c = bb.astype(np.int64)
        return c

    def entropy(self, vec):
        entropys = list()
        count = Counter(vec)
        countall = float(np.sum(list(count.values())))
        for item in count.items():
            counteach = item[1]
            prob = counteach / countall
            entropyeach = -prob * np.log(prob)
            entropys.append(entropyeach)
        entropy = np.sum(entropys)
        return entropy

    def show_aereport(self, my_ae, my_xtrain, my_ytrain, my_select):
        print("Report Creating Now...")
        digit_mean_ent, digit_x_ent, digit_img_dif, digit_x_dif = self.get_ent_diff(my_ae, my_xtrain, np.int(np.log2(my_ae.epoch_limit)) + 1)
        disp = "{0:.1f}".format(digit_mean_ent)
        disp2 = "{0:.1f}".format(digit_img_dif)
        print("LAST Entropy Diff = " + disp + "/" + disp2)

        self.report_header(my_ae)
        self.report_label(my_ae, my_select)
        self.report_mnist(my_ae, my_xtrain)
        self.report_train(my_ae, my_ytrain)
        self.report_calibration(my_ae, my_xtrain)
        self.report_w_x_hat(my_ae, my_xtrain, my_select)
        self.report_w_range(my_ae, my_select)
        self.report_b_range(my_ae, my_select)
        self.report_index1(my_ae)
        self.report_index2(my_ae, my_xtrain)
        self.report_index5(my_ae, my_ytrain)
        self.report_index3(my_ae)
        self.report_index4(my_ae)
        # Fig. Cost Volume
        self.draw_cost(my_ae, my_ytrain)
        self.draw_last_Wb_range(my_ae, my_xtrain)
        self.draw_last_Z_range(my_ae, my_xtrain)
        self.report_footer(my_ae, my_ytrain)

        if my_ae.dropout == 0.0:
            dout = "NoDrop"
        else:
            dout = "Drop" +  str(my_ae.dropout)
        if my_ae.noise == 0.0:
            noi = "NoNoise"
        else:
            noi = "Noise" +  str(my_ae.noise)

        # Save the report
        plt.savefig("./" + my_ae.exp_id + "_" + my_ae.func.name + "_Hidden" + str(my_ae.n_hidden) + "_" +
                    noi + "_" + dout + "_Epoch" + str(my_ae.epoch_limit) + "_SubScale" + str(my_ae.sub_scale) +
                    "_Wtran" + my_ae.Wtransport + ".png",
                    bbox_inches="tight", pad_inches=0.05)

        plt.show()


    # For Experiment Code
    def mkKernel(self, ks, sig, th, lm, ps):
        if not ks%2:
            exit(1)
        hks = ks/2
        theta = th * np.pi/180.
        psi = ps * np.pi/180.
        xs = np.linspace(-1., 1., ks)
        ys = np.linspace(-1., 1., ks)
        lmbd = np.float(lm)
        x, y = np.meshgrid(xs, ys)
        sigma = np.float(sig)/ks
        x_theta = x*np.cos(theta)+y*np.sin(theta)
        y_theta = -x*np.sin(theta)+y*np.cos(theta)
        return np.array(np.exp(-0.5*(x_theta**2+y_theta**2)/sigma**2)*np.cos(2.*np.pi*x_theta/lmbd + psi), dtype=np.float32)

    def mk_initfilter(self):
        self.initfilter = np.zeros((1, 784))

        kernel_size = 29
        pos_sigma = 4
        pos_lm = 50
        lm = 0.5 + pos_lm / 100.

        for i in range(20):
            for j in range(20):
                pos_th = 45 * (i + j * np.cos(i * np.pi))
                pos_psi = 160 * (i + j)
                filter = self.mkKernel(kernel_size, pos_sigma, pos_th, lm, pos_psi)
                corefilter = filter[10:19, 10:19]
                initfilter = np.zeros((28, 28))
                initfilter[i:i+9, j:j+9] = corefilter
                self.initfilter = np.concatenate([self.initfilter, initfilter.reshape(1,784)], axis=0)
