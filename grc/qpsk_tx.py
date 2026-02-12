#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: QPSK Tx
# Author: César
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import analog
from gnuradio import blocks
import pmt
from gnuradio import channels
from gnuradio.filter import firdes
from gnuradio import digital
from gnuradio import filter
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



class qpsk_tx(gr.top_block, Qt.QWidget):

    def __init__(self, freq=650e6, samp_rate_div=1, samp_sym=16, tx_file="../data/tx.txt", zmq_addr="tcp://0.0.0.0:18305"):
        gr.top_block.__init__(self, "QPSK Tx", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("QPSK Tx")
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

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "qpsk_tx")

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
        self.freq = freq
        self.samp_rate_div = samp_rate_div
        self.samp_sym = samp_sym
        self.tx_file = tx_file
        self.zmq_addr = zmq_addr

        ##################################################
        # Variables
        ##################################################
        self.max_sample_rate = max_sample_rate = 15e6
        self.samp_rate = samp_rate = max_sample_rate/samp_rate_div
        self.sym_rate = sym_rate = samp_rate/samp_sym
        self.beta = beta = 0.35
        self.rrc_taps = rrc_taps = firdes.root_raised_cosine(1.0, samp_rate,sym_rate, beta, samp_sym)
        self.qpsk = qpsk = digital.constellation_rect([0.707+0.707j, -0.707+0.707j, -0.707-0.707j, 0.707-0.707j], [0, 1, 2, 3],
        4, 2, 2, 1, 1).base()
        self.noise = noise = 0.15
        self.mod_ord = mod_ord = 2
        self.gain = gain = 85

        ##################################################
        # Blocks
        ##################################################

        self._noise_range = qtgui.Range(0, 1, 0.01, 0.15, 200)
        self._noise_win = qtgui.RangeWidget(self._noise_range, self.set_noise, "Sim: Noise Voltage", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._noise_win)
        self.zeromq_sub_source_0 = zeromq.sub_source(gr.sizeof_gr_complex, 1, zmq_addr, 5000, False, (-1), '', False)
        self.zeromq_pub_sink_0 = zeromq.pub_sink(gr.sizeof_gr_complex, 1, zmq_addr, 100, False, (-1), '', True, True)
        self.root_raised_cosine_filter_0 = filter.fir_filter_ccf(
            1,
            firdes.root_raised_cosine(
                1,
                samp_rate,
                (sym_rate*mod_ord),
                0.35,
                samp_sym))
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
        self.qtgui_freq_sink_x_0.set_y_label('Power', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(True)
        self.qtgui_freq_sink_x_0.set_fft_average(0.05)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(True)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(True)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["red", "red", "green", "black", "cyan",
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


        labels = ['Tx', 'Rx', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
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
        self._gain_range = qtgui.Range(0, 89.75, 0.25, 85, 200)
        self._gain_win = qtgui.RangeWidget(self._gain_range, self.set_gain, "Exp: Gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._gain_win)
        self.digital_symbol_sync_xx_0 = digital.symbol_sync_cc(
            digital.TED_EARLY_LATE,
            samp_sym,
            (2*np.pi/100),
            1.0,
            1.0,
            1.5,
            1,
            digital.constellation_qpsk().base(),
            digital.IR_MMSE_8TAP,
            128,
            [])
        self.digital_constellation_encoder_bc_0 = digital.constellation_encoder_bc(qpsk)
        self.channels_channel_model_0 = channels.channel_model(
            noise_voltage=noise,
            frequency_offset=0.0,
            epsilon=1.0,
            taps=[1.0],
            noise_seed=0,
            block_tags=False)
        self.blocks_throttle2_0 = blocks.throttle( gr.sizeof_gr_complex*1, samp_rate, True, 0 if "auto" == "auto" else max( int(float(0.1) * samp_rate) if "auto" == "time" else int(0.1), 1) )
        self.blocks_repeat_0 = blocks.repeat(gr.sizeof_gr_complex*1, samp_sym)
        self.blocks_pack_k_bits_bb_0 = blocks.pack_k_bits_bb(mod_ord)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_char*1, tx_file, True, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_add_const_vxx_0 = blocks.add_const_bb(208)
        self.analog_agc_xx_0 = analog.agc_cc((1e-4), 1.0, 1.0, 65536)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_agc_xx_0, 0), (self.digital_symbol_sync_xx_0, 0))
        self.connect((self.blocks_add_const_vxx_0, 0), (self.blocks_pack_k_bits_bb_0, 0))
        self.connect((self.blocks_file_source_0, 0), (self.blocks_add_const_vxx_0, 0))
        self.connect((self.blocks_pack_k_bits_bb_0, 0), (self.digital_constellation_encoder_bc_0, 0))
        self.connect((self.blocks_repeat_0, 0), (self.root_raised_cosine_filter_0, 0))
        self.connect((self.channels_channel_model_0, 0), (self.blocks_throttle2_0, 0))
        self.connect((self.channels_channel_model_0, 0), (self.zeromq_pub_sink_0, 0))
        self.connect((self.digital_constellation_encoder_bc_0, 0), (self.blocks_repeat_0, 0))
        self.connect((self.digital_symbol_sync_xx_0, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.root_raised_cosine_filter_0, 0), (self.channels_channel_model_0, 0))
        self.connect((self.zeromq_sub_source_0, 0), (self.analog_agc_xx_0, 0))
        self.connect((self.zeromq_sub_source_0, 0), (self.qtgui_freq_sink_x_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "qpsk_tx")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

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
        self.blocks_repeat_0.set_interpolation(self.samp_sym)
        self.digital_symbol_sync_xx_0.set_sps(self.samp_sym)
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.sym_rate*self.mod_ord), 0.35, self.samp_sym))

    def get_tx_file(self):
        return self.tx_file

    def set_tx_file(self, tx_file):
        self.tx_file = tx_file
        self.blocks_file_source_0.open(self.tx_file, True)

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
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.sym_rate*self.mod_ord), 0.35, self.samp_sym))

    def get_sym_rate(self):
        return self.sym_rate

    def set_sym_rate(self, sym_rate):
        self.sym_rate = sym_rate
        self.set_rrc_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.sym_rate, self.beta, self.samp_sym))
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.sym_rate*self.mod_ord), 0.35, self.samp_sym))

    def get_beta(self):
        return self.beta

    def set_beta(self, beta):
        self.beta = beta
        self.set_rrc_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.sym_rate, self.beta, self.samp_sym))

    def get_rrc_taps(self):
        return self.rrc_taps

    def set_rrc_taps(self, rrc_taps):
        self.rrc_taps = rrc_taps

    def get_qpsk(self):
        return self.qpsk

    def set_qpsk(self, qpsk):
        self.qpsk = qpsk
        self.digital_constellation_encoder_bc_0.set_constellation(self.qpsk)

    def get_noise(self):
        return self.noise

    def set_noise(self, noise):
        self.noise = noise
        self.channels_channel_model_0.set_noise_voltage(self.noise)

    def get_mod_ord(self):
        return self.mod_ord

    def set_mod_ord(self, mod_ord):
        self.mod_ord = mod_ord
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.sym_rate*self.mod_ord), 0.35, self.samp_sym))

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain



def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--freq", dest="freq", type=eng_float, default=eng_notation.num_to_str(float(650e6)),
        help="Set carrier frequency [default=%(default)r]")
    parser.add_argument(
        "--samp-rate-div", dest="samp_rate_div", type=intx, default=1,
        help="Set sample rate division [default=%(default)r]")
    parser.add_argument(
        "--samp-sym", dest="samp_sym", type=intx, default=16,
        help="Set samples per symbol [default=%(default)r]")
    parser.add_argument(
        "--tx-file", dest="tx_file", type=str, default="../data/tx.txt",
        help="Set input tx file [default=%(default)r]")
    parser.add_argument(
        "--zmq-addr", dest="zmq_addr", type=str, default="tcp://0.0.0.0:18305",
        help="Set ZMQ address [default=%(default)r]")
    return parser


def main(top_block_cls=qpsk_tx, options=None):
    if options is None:
        options = argument_parser().parse_args()

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(freq=options.freq, samp_rate_div=options.samp_rate_div, samp_sym=options.samp_sym, tx_file=options.tx_file, zmq_addr=options.zmq_addr)

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
