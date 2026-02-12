#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: QPSK Rx
# Author: César
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import analog
from gnuradio import blocks
from gnuradio import blocks, gr
from gnuradio import digital
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import zeromq
import numpy as np
import sip
import threading



class qpsk_rx(gr.top_block, Qt.QWidget):

    def __init__(self, file_rx="../data/qpsk.complex64", freq=650e6, samp_rate_div=1, samp_sym=16, zmq_addr="tcp://0.0.0.0:18305"):
        gr.top_block.__init__(self, "QPSK Rx", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("QPSK Rx")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "qpsk_rx")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Parameters
        ##################################################
        self.file_rx = file_rx
        self.freq = freq
        self.samp_rate_div = samp_rate_div
        self.samp_sym = samp_sym
        self.zmq_addr = zmq_addr

        ##################################################
        # Variables
        ##################################################
        self.max_sample_rate = max_sample_rate = 15e6
        self.samp_rate = samp_rate = max_sample_rate/samp_rate_div
        self.sym_rate = sym_rate = samp_rate/samp_sym
        self.qpsk = qpsk = digital.constellation_rect([0.707+0.707j, -0.707+0.707j, -0.707-0.707j, 0.707-0.707j], [0, 1, 2, 3],
        4, 2, 2, 1, 1).base()
        self.n_samples = n_samples = 128*1024
        self.beta = beta = 0.35
        self.rrc_taps = rrc_taps = firdes.root_raised_cosine(1.0, samp_rate,sym_rate, beta, samp_sym)
        self.n_skip = n_skip = int(n_samples*5)
        self.n_save = n_save = int(2.5*n_samples)
        self.mod_ord = mod_ord = 2
        self.gain = gain = 60
        self.eq_alg = eq_alg = digital.adaptive_algorithm_cma( qpsk, .0001, 1).base()

        ##################################################
        # Blocks
        ##################################################

        self.zeromq_sub_source_0 = zeromq.sub_source(gr.sizeof_gr_complex, 1, 'tcp://0.0.0.0:18305', 100, False, (-1), '', False)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            freq, #fc
            samp_rate, #bw
            "Spectrum", #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time((1/30))
        self.qtgui_freq_sink_x_0.set_y_axis((-100), 0)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(True)
        self.qtgui_freq_sink_x_0.set_fft_average(0.05)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(True)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(True)

        self.qtgui_freq_sink_x_0.disable_legend()


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [2, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["yellow", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_freq_sink_x_0_win, 0, 2, 2, 2)
        for r in range(0, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 4):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_edit_box_msg_0 = qtgui.edit_box_msg(qtgui.FLOAT, 'Waiting for value...', 'SNR:', False, True, '', None)
        self._qtgui_edit_box_msg_0_win = sip.wrapinstance(self.qtgui_edit_box_msg_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_edit_box_msg_0_win)
        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(
            1024, #size
            "Constellation", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0.set_update_time((1/30))
        self.qtgui_const_sink_x_0.set_y_axis((-2), 2)
        self.qtgui_const_sink_x_0.set_x_axis((-2), 2)
        self.qtgui_const_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0.enable_grid(True)
        self.qtgui_const_sink_x_0.enable_axis_labels(True)

        self.qtgui_const_sink_x_0.disable_legend()

        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["green", "red", "red", "red", "red",
            "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_const_sink_x_0_win, 0, 0, 2, 2)
        for r in range(0, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._gain_range = qtgui.Range(0, 76, 1, 60, 200)
        self._gain_win = qtgui.RangeWidget(self._gain_range, self.set_gain, "Exp: Gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._gain_win)
        self.freq_xlating_fir_filter_xxx_0 = filter.freq_xlating_fir_filter_ccc(1, rrc_taps, 0, samp_rate)
        self.digital_symbol_sync_xx_0 = digital.symbol_sync_cc(
            digital.TED_GARDNER,
            samp_sym,
            (2*np.pi/100),
            1,
            1,
            1.5,
            2,
            digital.constellation_bpsk().base(),
            digital.IR_PFB_NO_MF,
            64,
            [])
        self.digital_probe_mpsk_snr_est_c_0 = digital.probe_mpsk_snr_est_c(2, n_samples, 0.001)
        self.digital_mpsk_snr_est_cc_0 = digital.mpsk_snr_est_cc(2, 10000, 0.001)
        self.digital_linear_equalizer_0 = digital.linear_equalizer((2*samp_sym), 2, eq_alg, True, [ ], 'corr_est')
        self.digital_costas_loop_cc_0 = digital.costas_loop_cc((2*np.pi/100), (2**mod_ord), True)
        self.blocks_throttle2_0 = blocks.throttle( gr.sizeof_gr_complex*1, samp_rate, True, 0 if "auto" == "auto" else max( int(float(0.1) * samp_rate) if "auto" == "time" else int(0.1), 1) )
        self.blocks_skiphead_0 = blocks.skiphead(gr.sizeof_gr_complex*1, n_skip)
        self.blocks_message_debug_0 = blocks.message_debug(True, gr.log_levels.info)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, n_save)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, file_rx, False)
        self.blocks_file_sink_0.set_unbuffered(False)
        self.analog_agc2_xx_0 = analog.agc2_cc((1e-1), (1e-2), 3.0, 1.0, 65536)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.digital_probe_mpsk_snr_est_c_0, 'snr'), (self.qtgui_edit_box_msg_0, 'val'))
        self.msg_connect((self.qtgui_edit_box_msg_0, 'msg'), (self.blocks_message_debug_0, 'log'))
        self.connect((self.analog_agc2_xx_0, 0), (self.freq_xlating_fir_filter_xxx_0, 0))
        self.connect((self.blocks_head_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_skiphead_0, 0), (self.blocks_head_0, 0))
        self.connect((self.digital_costas_loop_cc_0, 0), (self.blocks_skiphead_0, 0))
        self.connect((self.digital_costas_loop_cc_0, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.digital_linear_equalizer_0, 0), (self.digital_mpsk_snr_est_cc_0, 0))
        self.connect((self.digital_linear_equalizer_0, 0), (self.digital_probe_mpsk_snr_est_c_0, 0))
        self.connect((self.digital_mpsk_snr_est_cc_0, 0), (self.digital_costas_loop_cc_0, 0))
        self.connect((self.digital_symbol_sync_xx_0, 0), (self.digital_linear_equalizer_0, 0))
        self.connect((self.freq_xlating_fir_filter_xxx_0, 0), (self.digital_symbol_sync_xx_0, 0))
        self.connect((self.zeromq_sub_source_0, 0), (self.analog_agc2_xx_0, 0))
        self.connect((self.zeromq_sub_source_0, 0), (self.blocks_throttle2_0, 0))
        self.connect((self.zeromq_sub_source_0, 0), (self.qtgui_freq_sink_x_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "qpsk_rx")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_file_rx(self):
        return self.file_rx

    def set_file_rx(self, file_rx):
        self.file_rx = file_rx
        self.blocks_file_sink_0.open(self.file_rx)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.qtgui_freq_sink_x_0.set_frequency_range(self.freq, self.samp_rate)

    def get_samp_rate_div(self):
        return self.samp_rate_div

    def set_samp_rate_div(self, samp_rate_div):
        self.samp_rate_div = samp_rate_div
        self.set_samp_rate(self.max_sample_rate/self.samp_rate_div)

    def get_samp_sym(self):
        return self.samp_sym

    def set_samp_sym(self, samp_sym):
        self.samp_sym = samp_sym
        self.set_rrc_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.sym_rate, self.beta, self.samp_sym))
        self.set_sym_rate(self.samp_rate/self.samp_sym)
        self.digital_symbol_sync_xx_0.set_sps(self.samp_sym)

    def get_zmq_addr(self):
        return self.zmq_addr

    def set_zmq_addr(self, zmq_addr):
        self.zmq_addr = zmq_addr

    def get_max_sample_rate(self):
        return self.max_sample_rate

    def set_max_sample_rate(self, max_sample_rate):
        self.max_sample_rate = max_sample_rate
        self.set_samp_rate(self.max_sample_rate/self.samp_rate_div)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_rrc_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.sym_rate, self.beta, self.samp_sym))
        self.set_sym_rate(self.samp_rate/self.samp_sym)
        self.blocks_throttle2_0.set_sample_rate(self.samp_rate)
        self.qtgui_freq_sink_x_0.set_frequency_range(self.freq, self.samp_rate)

    def get_sym_rate(self):
        return self.sym_rate

    def set_sym_rate(self, sym_rate):
        self.sym_rate = sym_rate
        self.set_rrc_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.sym_rate, self.beta, self.samp_sym))

    def get_qpsk(self):
        return self.qpsk

    def set_qpsk(self, qpsk):
        self.qpsk = qpsk

    def get_n_samples(self):
        return self.n_samples

    def set_n_samples(self, n_samples):
        self.n_samples = n_samples
        self.set_n_save(int(2.5*self.n_samples))
        self.set_n_skip(int(self.n_samples*5))
        self.digital_probe_mpsk_snr_est_c_0.set_msg_nsample(self.n_samples)

    def get_beta(self):
        return self.beta

    def set_beta(self, beta):
        self.beta = beta
        self.set_rrc_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.sym_rate, self.beta, self.samp_sym))

    def get_rrc_taps(self):
        return self.rrc_taps

    def set_rrc_taps(self, rrc_taps):
        self.rrc_taps = rrc_taps
        self.freq_xlating_fir_filter_xxx_0.set_taps(self.rrc_taps)

    def get_n_skip(self):
        return self.n_skip

    def set_n_skip(self, n_skip):
        self.n_skip = n_skip

    def get_n_save(self):
        return self.n_save

    def set_n_save(self, n_save):
        self.n_save = n_save
        self.blocks_head_0.set_length(self.n_save)

    def get_mod_ord(self):
        return self.mod_ord

    def set_mod_ord(self, mod_ord):
        self.mod_ord = mod_ord

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain

    def get_eq_alg(self):
        return self.eq_alg

    def set_eq_alg(self, eq_alg):
        self.eq_alg = eq_alg



def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--file-rx", dest="file_rx", type=str, default="../data/qpsk.complex64",
        help="Set output rx file [default=%(default)r]")
    parser.add_argument(
        "--freq", dest="freq", type=eng_float, default=eng_notation.num_to_str(float(650e6)),
        help="Set central frequency [default=%(default)r]")
    parser.add_argument(
        "--samp-rate-div", dest="samp_rate_div", type=intx, default=1,
        help="Set sample rate division [default=%(default)r]")
    parser.add_argument(
        "--samp-sym", dest="samp_sym", type=intx, default=16,
        help="Set samples per symbol [default=%(default)r]")
    parser.add_argument(
        "--zmq-addr", dest="zmq_addr", type=str, default="tcp://0.0.0.0:18305",
        help="Set ZMQ address [default=%(default)r]")
    return parser


def main(top_block_cls=qpsk_rx, options=None):
    if options is None:
        options = argument_parser().parse_args()

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(file_rx=options.file_rx, freq=options.freq, samp_rate_div=options.samp_rate_div, samp_sym=options.samp_sym, zmq_addr=options.zmq_addr)

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
