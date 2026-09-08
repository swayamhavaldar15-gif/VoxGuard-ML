import { useEffect, useRef, useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [speakerName, setSpeakerName] = useState("");
  const [registrationId, setRegistrationId] = useState("");

  const [recordingType, setRecordingType] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);

  const [originalAudio, setOriginalAudio] = useState(null);
  const [testAudio, setTestAudio] = useState(null);

  const [result, setResult] = useState(null);
  const [message, setMessage] = useState("");

  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  const [systemOnline, setSystemOnline] = useState(true);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const startRecording = async (type) => {
    try {
      setMessage("");
      setResult(null);

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      streamRef.current = stream;
      chunksRef.current = [];

      const recorder = new MediaRecorder(stream);

      mediaRecorderRef.current = recorder;
      setRecordingType(type);
      setIsRecording(true);
      setRecordingTime(0);

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, {
          type: "audio/webm",
        });

        if (type === "register") {
          setOriginalAudio(blob);
        } else {
          setTestAudio(blob);
        }

        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
        }

        setIsRecording(false);
        setRecordingType(null);
      };

      recorder.start();

      timerRef.current = setInterval(() => {
        setRecordingTime((previous) => {
          if (previous >= 4) {
            stopRecording();
            return 5;
          }

          return previous + 1;
        });
      }, 1000);
    } catch (error) {
      console.error(error);
      setMessage(
        "Microphone access denied. Please allow microphone permission."
      );
      setSystemOnline(false);
    }
  };

  const stopRecording = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      mediaRecorderRef.current.stop();
    }
  };

  const registerSpeaker = async () => {
    if (!speakerName.trim()) {
      setMessage("Please enter a speaker name.");
      return;
    }

    if (!originalAudio) {
      setMessage("Please record the speaker's voice first.");
      return;
    }

    try {
      setMessage("Registering voice identity...");
      setResult(null);

      const formData = new FormData();

      formData.append("audio", originalAudio, "registration.webm");
      formData.append("speaker_name", speakerName);

      const response = await axios.post(
        `${API_URL}/register-voice`,
        formData
      );

      const id =
        response.data.registration_id ||
        response.data.id ||
        response.data.registration?.id;

      setRegistrationId(String(id));

      setMessage(
        `Voice identity registered successfully. Registration ID: ${id}`
      );
    } catch (error) {
      console.error(error);

      setMessage(
        error.response?.data?.detail ||
          "Unable to register voice identity."
      );
    }
  };

  const verifySpeaker = async () => {
    if (!registrationId) {
      setMessage("Please register a speaker first.");
      return;
    }

    if (!testAudio) {
      setMessage("Please record a verification sample.");
      return;
    }

    try {
      setMessage("Running voice security analysis...");
      setResult(null);

      const formData = new FormData();

      formData.append("audio", testAudio, "verification.webm");
      formData.append("registration_id", registrationId);

      const response = await axios.post(
        `${API_URL}/verify-voice`,
        formData
      );

      setResult(response.data);
      setMessage("Security analysis completed.");
    } catch (error) {
      console.error(error);

      setMessage(
        error.response?.data?.detail ||
          "Voice verification failed."
      );
    }
  };

  const getRiskClass = () => {
    if (!result) return "neutral";

    const level = String(result.risk_level || "").toLowerCase();

    if (level.includes("high")) return "danger";
    if (level.includes("medium")) return "warning";
    return "safe";
  };

  const getSpeakerClass = () => {
    if (!result) return "neutral";

    const status = String(
      result.speaker_status || ""
    ).toLowerCase();

    if (status.includes("mismatch")) return "danger";
    if (status.includes("suspicious")) return "warning";
    if (status.includes("verified")) return "safe";

    return "neutral";
  };

  const getSpoofClass = () => {
    if (!result) return "neutral";

    const status = String(
      result.anti_spoof_status ||
        result.spoof_label ||
        ""
    ).toLowerCase();

    if (
      status.includes("spoof") ||
      status.includes("potential")
    ) {
      return "danger";
    }

    if (
      status.includes("bona") ||
      status.includes("genuine")
    ) {
      return "safe";
    }

    return "warning";
  };

  const riskScore = Number(result?.risk_score || 0);
  const similarity = Number(result?.similarity || 0);
  const spoofScore = Number(result?.spoof_score || 0);
  const bonaFideScore = Number(result?.bona_fide_score || 0);

  const similarityPercent =
    similarity <= 1 ? similarity * 100 : similarity;

  const spoofPercent =
    spoofScore <= 1 ? spoofScore * 100 : spoofScore;

  const bonaFidePercent =
    bonaFideScore <= 1
      ? bonaFideScore * 100
      : bonaFideScore;

  return (
    <div className="app">

      {/* BACKGROUND EFFECTS */}
      <div className="background-grid"></div>
      <div className="glow glow-one"></div>
      <div className="glow glow-two"></div>

      {/* NAVBAR */}
      <header className="navbar">

        <div className="brand">
          <div className="brand-icon">
            <span>◉</span>
          </div>

          <div>
            <div className="brand-name">
              VOX<span>GUARD</span>
            </div>

            <div className="brand-subtitle">
              AI VOICE SECURITY PLATFORM
            </div>
          </div>
        </div>

        <div className="nav-right">

          <div className="system-status">
            <span className="status-dot"></span>
            SYSTEM ONLINE
          </div>

          <div className="security-badge">
            <span>✦</span>
            AI PROTECTED
          </div>

        </div>
      </header>

      {/* HERO */}
      <main>

        <section className="hero">

          <div className="hero-left">

            <div className="eyebrow">
              <span></span>
              REAL-TIME VOICE INTELLIGENCE
            </div>

            <h1>
              Verify the voice.
              <br />
              <span>Protect the identity.</span>
            </h1>

            <p>
              VoxGuard combines speaker verification,
              anti-spoofing intelligence and risk analysis
              to detect suspicious voice activity in real time.
            </p>

            <div className="hero-features">

              <div>
                <span>01</span>
                SPEAKER
                <small>ECAPA-TDNN</small>
              </div>

              <div>
                <span>02</span>
                ANTI-SPOOF
                <small>AASIST AI</small>
              </div>

              <div>
                <span>03</span>
                RISK ENGINE
                <small>LIVE ANALYSIS</small>
              </div>

            </div>

          </div>

          <div className="hero-visual">

            <div className="orbital orbital-one"></div>
            <div className="orbital orbital-two"></div>

            <div className="voice-core">

              <div className="core-ring"></div>

              <div className="mic-symbol">
                🎙
              </div>

              <div className="core-label">
                VOICE
                <strong>INTELLIGENCE</strong>
              </div>

            </div>

            <div className="floating-card card-top">
              <span className="mini-icon">◈</span>
              <div>
                <small>ENGINE</small>
                <strong>ECAPA</strong>
              </div>
            </div>

            <div className="floating-card card-bottom">
              <span className="mini-icon">🛡</span>
              <div>
                <small>PROTECTION</small>
                <strong>AASIST</strong>
              </div>
            </div>

          </div>

        </section>

        {/* CONTROL PANEL */}
        <section className="workspace">

          <div className="section-heading">
            <div>
              <span className="section-number">01</span>
              VOICE SECURITY CONSOLE
            </div>

            <span className="live-label">
              ● LIVE
            </span>
          </div>

          <div className="console-grid">

            {/* REGISTRATION */}
            <div className="panel registration-panel">

              <div className="panel-header">

                <div className="panel-title">
                  <div className="panel-icon blue">
                    ID
                  </div>

                  <div>
                    <h3>Speaker Registration</h3>
                    <span>Create voice identity</span>
                  </div>
                </div>

                <span className="step-tag">
                  STEP 01
                </span>

              </div>

              <div className="input-group">
                <label>SPEAKER NAME</label>

                <input
                  type="text"
                  value={speakerName}
                  onChange={(e) =>
                    setSpeakerName(e.target.value)
                  }
                  placeholder="Enter speaker name"
                  disabled={isRecording}
                />
              </div>

              <div
                className={`record-box ${
                  isRecording && recordingType === "register"
                    ? "recording"
                    : ""
                }`}
              >

                <div className="record-visual">

                  <div className="wave wave-1"></div>
                  <div className="wave wave-2"></div>
                  <div className="wave wave-3"></div>
                  <div className="wave wave-4"></div>
                  <div className="wave wave-5"></div>

                </div>

                <div className="record-info">

                  <strong>
                    {isRecording &&
                    recordingType === "register"
                      ? "LISTENING..."
                      : originalAudio
                      ? "VOICE CAPTURED"
                      : "READY TO CAPTURE"}
                  </strong>

                  <span>
                    {isRecording &&
                    recordingType === "register"
                      ? `00:0${recordingTime}`
                      : "5 SECOND SAMPLE"}
                  </span>

                </div>

                <button
                  className={`record-button ${
                    isRecording &&
                    recordingType === "register"
                      ? "stop"
                      : ""
                  }`}
                  onClick={() =>
                    isRecording &&
                    recordingType === "register"
                      ? stopRecording()
                      : startRecording("register")
                  }
                >
                  {isRecording &&
                  recordingType === "register"
                    ? "■"
                    : "●"}
                </button>

              </div>

              <button
                className="primary-button"
                onClick={registerSpeaker}
                disabled={!originalAudio || isRecording}
              >
                <span>REGISTER VOICE IDENTITY</span>
                <span>→</span>
              </button>

              {registrationId && (
                <div className="registration-success">
                  <span>✓</span>
                  Identity registered · ID #{registrationId}
                </div>
              )}

            </div>

            {/* VERIFICATION */}
            <div className="panel verification-panel">

              <div className="panel-header">

                <div className="panel-title">

                  <div className="panel-icon purple">
                    VR
                  </div>

                  <div>
                    <h3>Voice Verification</h3>
                    <span>Analyze speaker authenticity</span>
                  </div>

                </div>

                <span className="step-tag">
                  STEP 02
                </span>

              </div>

              <div className="verify-target">

                <div className="target-avatar">
                  {speakerName
                    ? speakerName.charAt(0).toUpperCase()
                    : "?"}
                </div>

                <div>
                  <small>REGISTERED IDENTITY</small>
                  <strong>
                    {speakerName || "No speaker selected"}
                  </strong>
                </div>

                <div className="target-status">
                  {registrationId ? "READY" : "WAITING"}
                </div>

              </div>

              <div
                className={`record-box verify-record ${
                  isRecording && recordingType === "verify"
                    ? "recording"
                    : ""
                }`}
              >

                <div className="record-visual">

                  <div className="wave wave-1"></div>
                  <div className="wave wave-2"></div>
                  <div className="wave wave-3"></div>
                  <div className="wave wave-4"></div>
                  <div className="wave wave-5"></div>

                </div>

                <div className="record-info">

                  <strong>
                    {isRecording &&
                    recordingType === "verify"
                      ? "ANALYZING..."
                      : testAudio
                      ? "SAMPLE READY"
                      : "CAPTURE VERIFICATION"}
                  </strong>

                  <span>
                    {isRecording &&
                    recordingType === "verify"
                      ? `00:0${recordingTime}`
                      : "5 SECOND SAMPLE"}
                  </span>

                </div>

                <button
                  className={`record-button ${
                    isRecording &&
                    recordingType === "verify"
                      ? "stop"
                      : ""
                  }`}
                  onClick={() =>
                    isRecording &&
                    recordingType === "verify"
                      ? stopRecording()
                      : startRecording("verify")
                  }
                >
                  {isRecording &&
                  recordingType === "verify"
                    ? "■"
                    : "●"}
                </button>

              </div>

              <button
                className="primary-button purple-button"
                onClick={verifySpeaker}
                disabled={!testAudio || !registrationId || isRecording}
              >
                <span>RUN SECURITY ANALYSIS</span>
                <span>→</span>
              </button>

            </div>

          </div>

        </section>

        {/* MESSAGE */}
        {message && (
          <div className="message-bar">
            <span>●</span>
            {message}
          </div>
        )}

        {/* ANALYSIS RESULTS */}
        <section className="results-section">

          <div className="section-heading">
            <div>
              <span className="section-number">02</span>
              SECURITY ANALYSIS
            </div>

            {result && (
              <span className={`analysis-state ${getRiskClass()}`}>
                ANALYSIS COMPLETE
              </span>
            )}
          </div>

          {!result ? (

            <div className="empty-analysis">

              <div className="empty-icon">
                ◌
              </div>

              <h3>Awaiting Voice Analysis</h3>

              <p>
                Register a speaker and capture a verification
                sample to begin the VoxGuard security analysis.
              </p>

              <div className="analysis-pipeline">

                <span>VOICE INPUT</span>
                <b>→</b>
                <span>ECAPA</span>
                <b>→</b>
                <span>AASIST</span>
                <b>→</b>
                <span>RISK ENGINE</span>

              </div>

            </div>

          ) : (

            <div className="results-grid">

              {/* RISK SCORE */}
              <div className={`risk-card ${getRiskClass()}`}>

                <div className="result-card-header">
                  <span>OVERALL RISK</span>
                  <span>● LIVE</span>
                </div>

                <div className="risk-circle">

                  <svg viewBox="0 0 120 120">

                    <circle
                      className="risk-track"
                      cx="60"
                      cy="60"
                      r="50"
                    />

                    <circle
                      className="risk-progress"
                      cx="60"
                      cy="60"
                      r="50"
                      style={{
                        strokeDasharray: 314,
                        strokeDashoffset:
                          314 - (314 * riskScore) / 100,
                      }}
                    />

                  </svg>

                  <div className="risk-number">
                    <strong>{riskScore}</strong>
                    <span>/100</span>
                  </div>

                </div>

                <div className="risk-result">
                  <span>RISK LEVEL</span>
                  <strong>
                    {result.risk_level || "UNKNOWN"}
                  </strong>
                </div>

              </div>

              {/* SPEAKER */}
              <div className="metric-card">

                <div className="metric-header">
                  <div className="metric-icon blue">
                    ID
                  </div>

                  <div>
                    <small>SPEAKER VERIFICATION</small>
                    <strong>
                      {result.speaker_status ||
                        "UNKNOWN"}
                    </strong>
                  </div>
                </div>

                <div className="metric-value">
                  {similarityPercent.toFixed(1)}
                  <span>%</span>
                </div>

                <div className="progress-line">
                  <div
                    style={{
                      width: `${Math.min(
                        similarityPercent,
                        100
                      )}%`,
                    }}
                  ></div>
                </div>

                <div className="metric-footer">
                  <span>ECAPA similarity</span>

                  <b className={getSpeakerClass()}>
                    {result.speaker_status ||
                      "PENDING"}
                  </b>
                </div>

              </div>

              {/* AASIST */}
              <div className="metric-card">

                <div className="metric-header">

                  <div className="metric-icon purple">
                    AI
                  </div>

                  <div>
                    <small>ANTI-SPOOFING</small>
                    <strong>
                      AASIST ANALYSIS
                    </strong>
                  </div>

                </div>

                <div className="dual-metrics">

                  <div>
                    <span>BONA FIDE</span>
                    <strong>
                      {bonaFidePercent.toFixed(1)}%
                    </strong>
                  </div>

                  <div>
                    <span>SPOOF</span>
                    <strong>
                      {spoofPercent.toFixed(1)}%
                    </strong>
                  </div>

                </div>

                <div className="progress-line spoof-line">
                  <div
                    style={{
                      width: `${Math.min(
                        spoofPercent,
                        100
                      )}%`,
                    }}
                  ></div>
                </div>

                <div className="metric-footer">
                  <span>
                    AASIST authenticity signal
                  </span>

                  <b className={getSpoofClass()}>
                    {result.anti_spoof_status ||
                      result.spoof_label ||
                      "UNKNOWN"}
                  </b>
                </div>

              </div>

              {/* DECISION */}
              <div className="decision-card">

                <div className="decision-icon">
                  {getRiskClass() === "safe"
                    ? "✓"
                    : getRiskClass() === "danger"
                    ? "!"
                    : "?"}
                </div>

                <div>

                  <small>VOXGUARD DECISION</small>

                  <h3>
                    {result.speaker_status ||
                      "ANALYSIS COMPLETE"}
                  </h3>

                  <p>
                    {getRiskClass() === "safe"
                      ? "Voice characteristics appear consistent with the registered identity."
                      : getRiskClass() === "danger"
                      ? "Security indicators require immediate attention."
                      : "Some security indicators require additional verification."}
                  </p>

                </div>

              </div>

            </div>

          )}

        </section>

      </main>

      {/* FOOTER */}
      <footer>

        <div>
          <strong>VOXGUARD</strong>
          <span>AI-powered voice security</span>
        </div>

        <div className="footer-tech">
          ECAPA-TDNN
          <span>•</span>
          AASIST
          <span>•</span>
          RISK ENGINE
        </div>

        <div className="footer-status">
          <span></span>
          SECURE SESSION
        </div>

      </footer>

    </div>
  );
}

export default App;