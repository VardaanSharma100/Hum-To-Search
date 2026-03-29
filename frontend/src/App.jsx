import { useRef, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import AudioVisualizer from './components/AudioVisualizer';
import './index.css';

function App() {
  const [status, setStatus] = useState('idle'); // idle, recording, loading, success
  const [result, setResult] = useState(null);
  const [audioData, setAudioData] = useState(new Uint8Array(0));

  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceRef = useRef(null);
  const reqFrameRef = useRef(null);
  const chunksRef = useRef([]);

  const startRecording = async () => {
    setStatus('recording');
    setResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      analyserRef.current = audioContextRef.current.createAnalyser();
      sourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
      sourceRef.current.connect(analyserRef.current);

      analyserRef.current.fftSize = 256;
      const bufferLength = analyserRef.current.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const updateData = () => {
        if (analyserRef.current && status === 'recording') {
          analyserRef.current.getByteFrequencyData(dataArray);
          setAudioData([...dataArray]);
          reqFrameRef.current = requestAnimationFrame(updateData);
        }
      };

      updateData();

      mediaRecorderRef.current = new MediaRecorder(stream);
      mediaRecorderRef.current.ondataavailable = e => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        setStatus('loading');
        cancelAnimationFrame(reqFrameRef.current);
        const blob = new Blob(chunksRef.current, { type: 'audio/wav' });
        chunksRef.current = [];

        const formData = new FormData();
        formData.append("file", blob, "hum.wav");

        try {
          const res = await fetch("http://localhost:8000/", {
            method: "POST",
            body: formData
          });
          const json = await res.json();
          setResult(json.prediction || json.match); // accommodating prompt format if modified later
          setStatus('success');
        } catch (e) {
          console.error(e);
          setResult("Error Connecting to Server");
          setStatus('success');
        }
      };

      mediaRecorderRef.current.start();
    } catch (err) {
      console.error(err);
      setStatus('idle');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && status === 'recording') {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(t => t.stop());
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(console.error);
    }
  };

  return (
    <div className="w-screen h-screen bg-[#1A1A1A] text-white flex flex-col items-center justify-center relative overflow-hidden">
      <div className="absolute inset-0 z-0">
        <Canvas camera={{ position: [0, 0, 15], fov: 60 }}>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} intensity={2} />
          <AudioVisualizer status={status} audioData={audioData} />
        </Canvas>
      </div>

      <div className="z-10 flex flex-col items-center select-none pointer-events-auto mt-[40vh]">
        {status === 'idle' && (
          <button
            onClick={startRecording}
            className="group relative px-10 py-5 rounded-full bg-white text-black font-bold uppercase tracking-[0.2em] transition-all duration-500 ease-[cubic-bezier(0.25,1,0.5,1)] hover:scale-105 active:scale-95 cursor-pointer overflow-hidden shadow-[0_0_0_rgba(255,255,255,0)] hover:shadow-[0_0_30px_rgba(255,255,255,0.3)]"
          >
            <span className="relative z-10 transition-transform duration-500 group-hover:scale-110 inline-block">Start Humming</span>
            <div className="absolute inset-0 h-full w-full border-[1px] border-white rounded-full scale-[1.1] opacity-0 group-hover:scale-[1.3] group-hover:opacity-100 transition-all duration-700 ease-out pointer-events-none"></div>
          </button>
        )}

        {status === 'recording' && (
          <div className="relative">
            {/* Pulsing ring background */}
            <div className="absolute -inset-2 bg-red-500 rounded-full opacity-20 animate-ping"></div>
            <button
              onClick={stopRecording}
              className="relative px-10 py-5 rounded-full bg-transparent border-[3px] border-red-500 text-red-500 font-bold uppercase tracking-[0.2em] transition-all duration-500 ease-[cubic-bezier(0.25,1,0.5,1)] hover:bg-red-500 hover:text-white hover:shadow-[0_0_40px_rgba(239,68,68,0.5)] hover:border-transparent active:scale-95 cursor-pointer flex items-center gap-3 overflow-hidden group"
            >
              <span className="w-3 h-3 bg-red-500 rounded-full animate-pulse group-hover:bg-white inline-block"></span>
              <span className="relative z-10">Stop Recording</span>
            </button>
          </div>
        )}

        {status === 'loading' && (
          <p className="text-gray-400 font-mono tracking-widest uppercase animate-pulse">Processing Audio...</p>
        )}

        {status === 'success' && (
          <div className="text-center transform translate-y-[-20vh]">
            <h2 className="text-sm tracking-[0.3em] text-gray-400 mb-4 uppercase">Matched Song</h2>
            <h1 className="text-5xl font-mono mb-8 text-white">
              <a
                href="https://www.youtube.com/watch?v=iIWoYaJRryw&list=RDiIWoYaJRryw&start_radio=1"
                target="_blank"
                rel="noopener noreferrer"
                className="text-white no-underline"
              >
                {result}
              </a>
            </h1>
            <button
              onClick={() => setStatus('idle')}
              className="px-6 py-3 mt-4 rounded-full border border-gray-600 text-gray-300 font-mono text-sm transition-all duration-300 ease-out hover:bg-white hover:text-black hover:scale-105 active:scale-95 cursor-pointer"
            >
              Try Again
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
