import json
import os

import websockets
import requests
import numpy as np
import time
import soundfile
import asyncio
import pyaudio

# from paddlespeech.cli.log import # logger

class TextHttpHandler:
    def __init__(self, server_ip="127.0.0.1", port=8090):
        """Text http client request

        Args:
            server_ip (str, optional): the text server ip. Defaults to "127.0.0.1".
            port (int, optional): the text server port. Defaults to 8090.
        """
        super().__init__()
        self.server_ip = server_ip
        self.port = port
        if server_ip is None or port is None:
            self.url = None
        else:
            self.url = 'http://' + self.server_ip + ":" + str(
                self.port) + '/paddlespeech/text'
        # logger.info(f"endpoint: {self.url}")

    def run(self, text):
        """Call the text server to process the specific text

        Args:
            text (str): the text to be processed

        Returns:
            str: punctuation text
        """
        if self.server_ip is None or self.port is None:
            return text
        request = {
            "text": text,
        }
        try:
            res = requests.post(url=self.url, data=json.dumps(request))
            response_dict = res.json()
            punc_text = response_dict["result"]["punc_text"]
        except Exception as e:
            # logger.error(f"Call punctuation {self.url} occurs error")
            # logger.error(e)
            punc_text = text

        return punc_text


class ASRWsAudioHandler:
    def __init__(self,
                 url=None,
                 port=None,
                 endpoint="/paddlespeech/asr/streaming",
                 punc_server_ip=None,
                 punc_server_port=None):
        """PaddleSpeech Online ASR Server Client  audio handler
           Online asr server use the websocket protocal
        Args:
            url (str, optional): the server ip. Defaults to None.
            port (int, optional): the server port. Defaults to None.
            endpoint(str, optional): to compatiable with python server and c++ server.
            punc_server_ip(str, optional): the punctuation server ip. Defaults to None.
            punc_server_port(int, optional): the punctuation port. Defaults to None
        """
        self.url = url
        self.port = port
        if url is None or port is None or endpoint is None:
            self.url = None
        else:
            self.url = "ws://" + self.url + ":" + str(self.port) + endpoint
        self.punc_server = TextHttpHandler(punc_server_ip, punc_server_port)
        # logger.info(f"endpoint: {self.url}")

    def read_wave(self, wavfile_path: str):
        """read the audio file from specific wavfile path

        Args:
            wavfile_path (str): the audio wavfile,
                                 we assume that audio sample rate matches the model

        Yields:
            numpy.array: the samall package audio pcm data
        """
        samples, sample_rate = soundfile.read(wavfile_path, dtype='int16')
        x_len = len(samples)
        assert sample_rate == 16000

        chunk_size = int(85 * sample_rate / 1000)  # 85ms, sample_rate = 16kHz

        if x_len % chunk_size != 0:
            padding_len_x = chunk_size - x_len % chunk_size
        else:
            padding_len_x = 0

        padding = np.zeros((padding_len_x), dtype=samples.dtype)
        padded_x = np.concatenate([samples, padding], axis=0)

        assert (x_len + padding_len_x) % chunk_size == 0
        num_chunk = (x_len + padding_len_x) / chunk_size
        num_chunk = int(num_chunk)
        for i in range(0, num_chunk):
            start = i * chunk_size
            end = start + chunk_size
            x_chunk = padded_x[start:end]
            yield x_chunk

    async def run(self, wavfile_path: str):
        """Send a audio file to online server

        Args:
            wavfile_path (str): audio path

        Returns:
            str: the final asr result
        """
        # # logger.debug("send a message to the server")

        if self.url is None:
            # logger.error("No asr server, please input valid ip and port")
            return ""

        # 1. send websocket handshake protocal
        start_time = time.time()
        async with websockets.connect(self.url) as ws:
            # 2. server has already received handshake protocal
            # client start to send the command
            audio_info = json.dumps(
                {
                    "name": "test.wav",
                    "signal": "start",
                    "nbest": 1
                },
                sort_keys=True,
                indent=4,
                separators=(',', ': '))
            await ws.send(audio_info)
            msg = await ws.recv()
            # logger.info("client receive msg={}".format(msg))

            # 3. send chunk audio data to engine
            for chunk_data in self.read_wave(wavfile_path):
                await ws.send(chunk_data.tobytes())
                msg = await ws.recv()
                msg = json.loads(msg)
                msg_out = msg
                # logger.info("client receive msg={}".format(msg))
            #client start to punctuation restore
            if self.punc_server and len(msg['result']) > 0:
                msg["result"] = self.punc_server.run(msg["result"])
                # logger.info("client punctuation restored msg={}".format(msg))
            # 4. we must send finished signal to the server
            audio_info = json.dumps(
                {
                    "name": "test.wav",
                    "signal": "end",
                    "nbest": 1
                },
                sort_keys=True,
                indent=4,
                separators=(',', ': '))
            await ws.send(audio_info)
            msg = await ws.recv()

            # 5. decode the bytes to str
            msg = json.loads(msg)

            if self.punc_server:
                msg["result"] = self.punc_server.run(msg["result"])

            # 6. logging the final result and comptute the statstics
            elapsed_time = time.time() - start_time
            audio_info = soundfile.info(wavfile_path)
            # logger.info("client final receive msg={}".format(msg))
            print(
                f"audio duration: {audio_info.duration}, elapsed time: {elapsed_time}, RTF={elapsed_time/audio_info.duration}"
            )

            result = msg

            return result

    async def run_online(self):
        """Send a audio file to online server

        Returns:
            str: the final asr result
        """
        # # logger.debug("send a message to the server")

        if self.url is None:
            # logger.error("No asr server, please input valid ip and port")
            return ""

        # 1. send websocket handshake protocal
        start_time = time.time()
        async with websockets.connect(self.url) as ws:
            # 2. server has already received handshake protocal
            # client start to send the command
            audio_info = json.dumps(
                {
                    "name": "test.wav",
                    "signal": "start",
                    "nbest": 1
                },
                sort_keys=True,
                indent=4,
                separators=(',', ': '))
            await ws.send(audio_info)
            msg = await ws.recv()
            # logger.info("client receive msg={}".format(msg))

            record_seconds = 0.085  # 录制时长/秒
            pformat = pyaudio.paInt16
            channels = 1
            rate = 16000  # 采样率/Hz

            audio = pyaudio.PyAudio()
            stream = audio.open(format=pformat,
                                channels=channels,
                                rate=rate,
                                input=True)


            while True:
                wav_data = stream.read(int(rate * record_seconds))
                # with wave.open(f"tmp_{count}.wav", "wb") as wf:
                #     wf.setnchannels(channels)
                #     wf.setsampwidth(pyaudio.get_sample_size(pformat))
                #     wf.setframerate(rate)
                #     wf.writeframes(wav_data)

                await ws.send(wav_data)
                msg = await ws.recv()
                msg = json.loads(msg)
                # print("client receive msg={}".format(msg))
                #client start to punctuation restore
                if self.punc_server and len(msg['result']) > 0:
                    msg["result"] = self.punc_server.run(msg["result"])
                print("client punctuation restored msg={}".format(msg))
            # 4. we must send finished signal to the server
            audio_info = json.dumps(
            {
                "name": "test.wav",
                "signal": "end",
                "nbest": 1
            },
            sort_keys=True,
            indent=4,
            separators=(',', ': ')
            )
            await ws.send(audio_info)
            msg = await ws.recv()

            # 5. decode the bytes to str
            msg = json.loads(msg)

            if self.punc_server:
                msg["result"] = self.punc_server.run(msg["result"])

            # 6. logging the final result and comptute the statstics
            elapsed_time = time.time() - start_time
            audio_info = soundfile.info(wavfile_path)
            # logger.info("client final receive msg={}".format(msg))
            print(
                f"audio duration: {audio_info.duration}, elapsed time: {elapsed_time}, RTF={elapsed_time/audio_info.duration}"
            )

            stream.stop_stream()
            stream.close()
            audio.terminate()

            result = msg

            return result

if __name__ == "__main__":
    server_ip = "192.168.211.48"
    port = "8090"
    punc_server_ip = "192.168.211.48"
    punc_server_port = "8190"
    wavfile = "16k16bit.wav"

    handler = ASRWsAudioHandler(
    server_ip,
    port,
    punc_server_ip=punc_server_ip,
    punc_server_port=punc_server_port)

    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(handler.run_online())













