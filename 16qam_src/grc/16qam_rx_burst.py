#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: 16-QAM Rx Burst (Headless Capture)
# Author: César
# Description: 16-QAM RoF Receiver (qam16_rx_burst)
# GNU Radio version: 3.10.12.0

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
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
import numpy as np
import threading




class qam16_rx_burst(gr.top_block):

    def __init__(self, gain=40, max_sample_rate=61.44e6, file_rx="./data/16qam.complex64", freq=650e6, samp_rate_div=1, samp_sym=4, args="", zmq_addr="tcp://0.0.0.0:18305"):
        gr.top_block.__init__(self, "16-QAM Rx Burst (Headless Capture)", catch_exceptions=True)
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
        self.zmq_addr = zmq_addr

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = max_sample_rate/samp_rate_div
        self.sym_rate = sym_rate = samp_rate/samp_sym
        self.qam16 = qam16 = digital.constellation_calcdist([-0.9486833-0.9486833j, -0.9486833-0.31622777j, -0.9486833+0.9486833j, -0.9486833+0.31622777j, -0.31622777-0.9486833j, -0.31622777-0.31622777j, -0.31622777+0.9486833j, -0.31622777+0.31622777j, 0.9486833-0.9486833j, 0.9486833-0.31622777j, 0.9486833+0.9486833j, 0.9486833+0.31622777j, 0.31622777-0.9486833j, 0.31622777-0.31622777j, 0.31622777+0.9486833j, 0.31622777+0.31622777j], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        4, 1, digital.constellation.POWER_NORMALIZATION).base()
        self.qam16.set_npwr(1.0)
        self.n_samples = n_samples = 128*1024
        self.beta = beta = 0.35
        self.rrc_taps = rrc_taps = firdes.root_raised_cosine(1.0, samp_rate,sym_rate, beta, (11*samp_sym+1))
        self.n_skip = n_skip = int(n_samples*5)
        self.n_save = n_save = int(2.5*n_samples)
        self.mod_ord = mod_ord = 4
        self.eq_alg = eq_alg = digital.adaptive_algorithm_cma( qam16, .0001, 1).base()

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
        self.freq_xlating_fir_filter_xxx_0 = filter.freq_xlating_fir_filter_ccf(1, rrc_taps, 0, samp_rate)
        self.digital_symbol_sync_xx_0 = digital.symbol_sync_cc(
            digital.TED_GARDNER,
            samp_sym,
            (2*np.pi/100),
            1,
            1,
            1.5,
            2,
            qam16,
            digital.IR_PFB_NO_MF,
            64,
            [])
        self.digital_probe_mpsk_snr_est_c_0 = digital.probe_mpsk_snr_est_c(2, (int(n_save/10)), 0.001)
        self.digital_linear_equalizer_0 = digital.linear_equalizer(15, 2, eq_alg, True, [ ], 'corr_est')
        self.digital_constellation_receiver_cb_0 = digital.constellation_receiver_cb(qam16, (2*np.pi/200), (-0.25), 0.25)
        self.digital_constellation_encoder_bc_snr = digital.constellation_encoder_bc(qam16)
        self.blocks_sub_xx_0 = blocks.sub_cc(1)
        self.blocks_skiphead_snr = blocks.skiphead(gr.sizeof_gr_complex*1, n_skip)
        self.blocks_skiphead_0 = blocks.skiphead(gr.sizeof_gr_complex*1, n_skip)
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_float*1)
        self.blocks_message_debug_0 = blocks.message_debug(True, gr.log_levels.info)
        self.blocks_head_snr = blocks.head(gr.sizeof_gr_complex*1, n_save)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, n_save)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, file_rx, False)
        self.blocks_file_sink_0.set_unbuffered(False)
        self.blocks_add_const_vxx_snr = blocks.add_const_cc(1.0+0j)
        self.analog_agc2_xx_0 = analog.agc2_cc((1e-1), (1e-2), 1.0, 1.0, 65536)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.digital_probe_mpsk_snr_est_c_0, 'snr'), (self.blocks_message_debug_0, 'log'))
        self.connect((self.analog_agc2_xx_0, 0), (self.freq_xlating_fir_filter_xxx_0, 0))
        self.connect((self.blocks_add_const_vxx_snr, 0), (self.blocks_skiphead_snr, 0))
        self.connect((self.blocks_head_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_head_snr, 0), (self.digital_probe_mpsk_snr_est_c_0, 0))
        self.connect((self.blocks_skiphead_0, 0), (self.blocks_head_0, 0))
        self.connect((self.blocks_skiphead_snr, 0), (self.blocks_head_snr, 0))
        self.connect((self.blocks_sub_xx_0, 0), (self.blocks_add_const_vxx_snr, 0))
        self.connect((self.digital_constellation_encoder_bc_snr, 0), (self.blocks_sub_xx_0, 1))
        self.connect((self.digital_constellation_receiver_cb_0, 1), (self.blocks_null_sink_0, 0))
        self.connect((self.digital_constellation_receiver_cb_0, 3), (self.blocks_null_sink_0, 2))
        self.connect((self.digital_constellation_receiver_cb_0, 2), (self.blocks_null_sink_0, 1))
        self.connect((self.digital_constellation_receiver_cb_0, 4), (self.blocks_skiphead_0, 0))
        self.connect((self.digital_constellation_receiver_cb_0, 4), (self.blocks_sub_xx_0, 0))
        self.connect((self.digital_constellation_receiver_cb_0, 0), (self.digital_constellation_encoder_bc_snr, 0))
        self.connect((self.digital_linear_equalizer_0, 0), (self.digital_constellation_receiver_cb_0, 0))
        self.connect((self.digital_symbol_sync_xx_0, 0), (self.digital_linear_equalizer_0, 0))
        self.connect((self.freq_xlating_fir_filter_xxx_0, 0), (self.digital_symbol_sync_xx_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.analog_agc2_xx_0, 0))


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
        self.set_rrc_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.sym_rate, self.beta, (11*self.samp_sym+1)))
        self.set_sym_rate(self.samp_rate/self.samp_sym)
        self.digital_symbol_sync_xx_0.set_sps(self.samp_sym)

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
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_sym_rate(self):
        return self.sym_rate

    def set_sym_rate(self, sym_rate):
        self.sym_rate = sym_rate
        self.set_rrc_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.sym_rate, self.beta, (11*self.samp_sym+1)))

    def get_qam16(self):
        return self.qam16

    def set_qam16(self, qam16):
        self.qam16 = qam16
        self.digital_constellation_encoder_bc_snr.set_constellation(self.qam16)

    def get_n_samples(self):
        return self.n_samples

    def set_n_samples(self, n_samples):
        self.n_samples = n_samples
        self.set_n_save(int(2.5*self.n_samples))
        self.set_n_skip(int(self.n_samples*5))

    def get_beta(self):
        return self.beta

    def set_beta(self, beta):
        self.beta = beta
        self.set_rrc_taps(firdes.root_raised_cosine(1.0, self.samp_rate, self.sym_rate, self.beta, (11*self.samp_sym+1)))

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
        self.digital_probe_mpsk_snr_est_c_0.set_msg_nsample((int(self.n_save/10)))
        self.blocks_head_snr.set_length(self.n_save)

    def get_mod_ord(self):
        return self.mod_ord

    def set_mod_ord(self, mod_ord):
        self.mod_ord = mod_ord

    def get_eq_alg(self):
        return self.eq_alg

    def set_eq_alg(self, eq_alg):
        self.eq_alg = eq_alg



def argument_parser():
    description = '16-QAM RoF Receiver (qam16_rx_burst)'
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--gain", dest="gain", type=eng_float, default=eng_notation.num_to_str(float(40)),
        help="Set RX gain (dB) [default=%(default)r]")
    parser.add_argument(
        "--max-sample-rate", dest="max_sample_rate", type=eng_float, default=eng_notation.num_to_str(float(61.44e6)),
        help="Set max sample rate [default=%(default)r]")
    parser.add_argument(
        "--file-rx", dest="file_rx", type=str, default="./data/16qam.complex64",
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
    parser.add_argument(
        "--zmq-addr", dest="zmq_addr", type=str, default="tcp://0.0.0.0:18305",
        help="Set ZMQ address [default=%(default)r]")
    return parser


def main(top_block_cls=qam16_rx_burst, options=None):
    if options is None:
        options = argument_parser().parse_args()
    tb = top_block_cls(gain=options.gain, max_sample_rate=options.max_sample_rate, file_rx=options.file_rx, freq=options.freq, samp_rate_div=options.samp_rate_div, samp_sym=options.samp_sym, args=options.args, zmq_addr=options.zmq_addr)

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
