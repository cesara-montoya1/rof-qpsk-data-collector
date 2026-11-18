#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: QPSK Rx v2 Burst Raw I/Q
# Author: César
# Description: Raw I/Q Burst Capture (qpsk_rx_v2_burst_raw) - 100% Offline DSP
# GNU Radio version: 3.10.12.0

from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
import numpy as np
import threading




class qpsk_rx_v2_burst_raw(gr.top_block):

    def __init__(self, gain=40, max_sample_rate=61.44e6, file_rx="./data/qpsk_raw.complex64", freq=650e6, samp_rate_div=1, samp_sym=4, args=""):
        gr.top_block.__init__(self, "QPSK Rx v2 Burst Raw I/Q", catch_exceptions=True)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Parameters
        ##################################################
        self.gain = gain
        self.max_sample_rate = max_sample_rate
        self.file_rx = file_rx
        self.freq = freq
        self.samp_rate_div = samp_rate_div
        self.samp_sym = samp_sym
        self.args = args

        ##################################################
        # Variables
        ##################################################
        self.n_samples = n_samples = 128*1024
        self.samp_rate = samp_rate = max_sample_rate/samp_rate_div
        self.n_skip = n_skip = int(n_samples*5)
        self.n_save = n_save = int(2.5*n_samples)

        ##################################################
        # Blocks
        ##################################################

        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join((args, "num_recv_frames=1024")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_time_unknown_pps(uhd.time_spec(0))

        self.uhd_usrp_source_0.set_center_freq(freq, 0)
        self.uhd_usrp_source_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_source_0.set_gain(gain, 0)
        self.blocks_skiphead_0 = blocks.skiphead(gr.sizeof_gr_complex*1, (n_skip * samp_sym))
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, (n_save * samp_sym))
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, file_rx, False)
        self.blocks_file_sink_0.set_unbuffered(False)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_head_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_skiphead_0, 0), (self.blocks_head_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_skiphead_0, 0))


    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.uhd_usrp_source_0.set_gain(self.gain, 0)

    def get_max_sample_rate(self):
        return self.max_sample_rate

    def set_max_sample_rate(self, max_sample_rate):
        self.max_sample_rate = max_sample_rate
        self.set_samp_rate(self.max_sample_rate/self.samp_rate_div)

    def get_file_rx(self):
        return self.file_rx

    def set_file_rx(self, file_rx):
        self.file_rx = file_rx
        self.blocks_file_sink_0.open(self.file_rx)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.uhd_usrp_source_0.set_center_freq(self.freq, 0)

    def get_samp_rate_div(self):
        return self.samp_rate_div

    def set_samp_rate_div(self, samp_rate_div):
        self.samp_rate_div = samp_rate_div
        self.set_samp_rate(self.max_sample_rate/self.samp_rate_div)

    def get_samp_sym(self):
        return self.samp_sym

    def set_samp_sym(self, samp_sym):
        self.samp_sym = samp_sym
        self.blocks_head_0.set_length((self.n_save * self.samp_sym))

    def get_args(self):
        return self.args

    def set_args(self, args):
        self.args = args

    def get_n_samples(self):
        return self.n_samples

    def set_n_samples(self, n_samples):
        self.n_samples = n_samples
        self.set_n_save(int(2.5*self.n_samples))
        self.set_n_skip(int(self.n_samples*5))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_n_skip(self):
        return self.n_skip

    def set_n_skip(self, n_skip):
        self.n_skip = n_skip

    def get_n_save(self):
        return self.n_save

    def set_n_save(self, n_save):
        self.n_save = n_save
        self.blocks_head_0.set_length((self.n_save * self.samp_sym))



def argument_parser():
    description = 'Raw I/Q Burst Capture (qpsk_rx_v2_burst_raw) - 100% Offline DSP'
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--gain", dest="gain", type=eng_float, default=eng_notation.num_to_str(float(40)),
        help="Set RX gain (dB) [default=%(default)r]")
    parser.add_argument(
        "--max-sample-rate", dest="max_sample_rate", type=eng_float, default=eng_notation.num_to_str(float(61.44e6)),
        help="Set max sample rate [default=%(default)r]")
    parser.add_argument(
        "--file-rx", dest="file_rx", type=str, default="./data/qpsk_raw.complex64",
        help="Set output rx file [default=%(default)r]")
    parser.add_argument(
        "--freq", dest="freq", type=eng_float, default=eng_notation.num_to_str(float(650e6)),
        help="Set central frequency [default=%(default)r]")
    parser.add_argument(
        "--samp-rate-div", dest="samp_rate_div", type=intx, default=1,
        help="Set sample rate division [default=%(default)r]")
    parser.add_argument(
        "--samp-sym", dest="samp_sym", type=intx, default=4,
        help="Set samples per symbol [default=%(default)r]")
    parser.add_argument(
        "--args", dest="args", type=str, default="",
        help="Set USRP device args [default=%(default)r]")
    return parser


def main(top_block_cls=qpsk_rx_v2_burst_raw, options=None):
    if options is None:
        options = argument_parser().parse_args()
    tb = top_block_cls(gain=options.gain, max_sample_rate=options.max_sample_rate, file_rx=options.file_rx, freq=options.freq, samp_rate_div=options.samp_rate_div, samp_sym=options.samp_sym, args=options.args)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.flowgraph_started.set()

    tb.wait()


if __name__ == '__main__':
    main()
