#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: 16-QAM Tx Burst (Headless)
# Author: César
# Description: 16-QAM RoF Transmitter (qam16_tx_burst)
# GNU Radio version: 3.10.12.0

from gnuradio import blocks
import pmt
from gnuradio import digital
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
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




class qam16_tx_burst(gr.top_block):

    def __init__(self, gain=85, max_sample_rate=61.44e6, file_tx="../data/tx.txt", freq=650e6, samp_rate_div=1, samp_sym=4, args="", zmq_addr="tcp://0.0.0.0:18305"):
        gr.top_block.__init__(self, "16-QAM Tx Burst (Headless)", catch_exceptions=True)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Parameters
        ##################################################
        self.gain = gain
        self.max_sample_rate = max_sample_rate
        self.file_tx = file_tx
        self.freq = freq
        self.samp_rate_div = samp_rate_div
        self.samp_sym = samp_sym
        self.args = args
        self.zmq_addr = zmq_addr

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = max_sample_rate/samp_rate_div
        self.sym_rate = sym_rate = samp_rate/samp_sym
        self.beta = beta = 0.35
        self.rrc_taps = rrc_taps = firdes.root_raised_cosine(1.0, samp_rate,sym_rate, beta, (11*samp_sym+1))
        self.qam16 = qam16 = digital.constellation_calcdist([-0.9486833-0.9486833j, -0.9486833-0.31622777j, -0.9486833+0.9486833j, -0.9486833+0.31622777j, -0.31622777-0.9486833j, -0.31622777-0.31622777j, -0.31622777+0.9486833j, -0.31622777+0.31622777j, 0.9486833-0.9486833j, 0.9486833-0.31622777j, 0.9486833+0.9486833j, 0.9486833+0.31622777j, 0.31622777-0.9486833j, 0.31622777-0.31622777j, 0.31622777+0.9486833j, 0.31622777+0.31622777j], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        4, 1, digital.constellation.POWER_NORMALIZATION).base()
        self.qam16.set_npwr(1.0)
        self.mod_ord = mod_ord = 4

        ##################################################
        # Blocks
        ##################################################

        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join((args, "num_send_frames=1024")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
            "",
        )
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        # No synchronization enforced.

        self.uhd_usrp_sink_0.set_center_freq(freq, 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0.set_gain(gain, 0)
        self.root_raised_cosine_filter_0 = filter.interp_fir_filter_ccf(
            samp_sym,
            firdes.root_raised_cosine(
                (0.5*samp_sym),
                samp_rate,
                sym_rate,
                beta,
                (11*samp_sym+1)))
        self.digital_constellation_encoder_bc_0 = digital.constellation_encoder_bc(qam16)
        self.blocks_pack_k_bits_bb_0 = blocks.pack_k_bits_bb(mod_ord)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_char*1, file_tx, True, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_add_const_vxx_0 = blocks.add_const_bb(208)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_add_const_vxx_0, 0), (self.blocks_pack_k_bits_bb_0, 0))
        self.connect((self.blocks_file_source_0, 0), (self.blocks_add_const_vxx_0, 0))
        self.connect((self.blocks_pack_k_bits_bb_0, 0), (self.digital_constellation_encoder_bc_0, 0))
        self.connect((self.digital_constellation_encoder_bc_0, 0), (self.root_raised_cosine_filter_0, 0))
        self.connect((self.root_raised_cosine_filter_0, 0), (self.uhd_usrp_sink_0, 0))


    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.uhd_usrp_sink_0.set_gain(self.gain, 0)

    def get_max_sample_rate(self):
        return self.max_sample_rate

    def set_max_sample_rate(self, max_sample_rate):
        self.max_sample_rate = max_sample_rate
        self.set_samp_rate(self.max_sample_rate/self.samp_rate_div)

    def get_file_tx(self):
        return self.file_tx

    def set_file_tx(self, file_tx):
        self.file_tx = file_tx
        self.blocks_file_source_0.open(self.file_tx, True)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.uhd_usrp_sink_0.set_center_freq(self.freq, 0)

    def get_samp_rate_div(self):
        return self.samp_rate_div

    def set_samp_rate_div(self, samp_rate_div):
        self.samp_rate_div = samp_rate_div
        self.set_samp_rate(self.max_sample_rate/self.samp_rate_div)

    def get_samp_sym(self):
        return self.samp_sym

    def set_samp_sym(self, samp_sym):
        self.samp_sym = samp_sym
        self.set_rrc_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.sym_rate, self.beta, (11*self.samp_sym+1)))
        self.set_sym_rate(self.samp_rate/self.samp_sym)
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine((0.5*self.samp_sym), self.samp_rate, self.sym_rate, self.beta, (11*self.samp_sym+1)))

    def get_args(self):
        return self.args

    def set_args(self, args):
        self.args = args

    def get_zmq_addr(self):
        return self.zmq_addr

    def set_zmq_addr(self, zmq_addr):
        self.zmq_addr = zmq_addr

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_rrc_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.sym_rate, self.beta, (11*self.samp_sym+1)))
        self.set_sym_rate(self.samp_rate/self.samp_sym)
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine((0.5*self.samp_sym), self.samp_rate, self.sym_rate, self.beta, (11*self.samp_sym+1)))
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)

    def get_sym_rate(self):
        return self.sym_rate

    def set_sym_rate(self, sym_rate):
        self.sym_rate = sym_rate
        self.set_rrc_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.sym_rate, self.beta, (11*self.samp_sym+1)))
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine((0.5*self.samp_sym), self.samp_rate, self.sym_rate, self.beta, (11*self.samp_sym+1)))

    def get_beta(self):
        return self.beta

    def set_beta(self, beta):
        self.beta = beta
        self.set_rrc_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.sym_rate, self.beta, (11*self.samp_sym+1)))
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine((0.5*self.samp_sym), self.samp_rate, self.sym_rate, self.beta, (11*self.samp_sym+1)))

    def get_rrc_taps(self):
        return self.rrc_taps

    def set_rrc_taps(self, rrc_taps):
        self.rrc_taps = rrc_taps

    def get_qam16(self):
        return self.qam16

    def set_qam16(self, qam16):
        self.qam16 = qam16
        self.digital_constellation_encoder_bc_0.set_constellation(self.qam16)

    def get_mod_ord(self):
        return self.mod_ord

    def set_mod_ord(self, mod_ord):
        self.mod_ord = mod_ord



def argument_parser():
    description = '16-QAM RoF Transmitter (qam16_tx_burst)'
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--gain", dest="gain", type=eng_float, default=eng_notation.num_to_str(float(85)),
        help="Set TX gain (dB) [default=%(default)r]")
    parser.add_argument(
        "--max-sample-rate", dest="max_sample_rate", type=eng_float, default=eng_notation.num_to_str(float(61.44e6)),
        help="Set max sample rate [default=%(default)r]")
    parser.add_argument(
        "--file-tx", dest="file_tx", type=str, default="../data/tx.txt",
        help="Set input tx file [default=%(default)r]")
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
    parser.add_argument(
        "--zmq-addr", dest="zmq_addr", type=str, default="tcp://0.0.0.0:18305",
        help="Set ZMQ address [default=%(default)r]")
    return parser


def main(top_block_cls=qam16_tx_burst, options=None):
    if options is None:
        options = argument_parser().parse_args()
    tb = top_block_cls(gain=options.gain, max_sample_rate=options.max_sample_rate, file_tx=options.file_tx, freq=options.freq, samp_rate_div=options.samp_rate_div, samp_sym=options.samp_sym, args=options.args, zmq_addr=options.zmq_addr)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.flowgraph_started.set()

    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
