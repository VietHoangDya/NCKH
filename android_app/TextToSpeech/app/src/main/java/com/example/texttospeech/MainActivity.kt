package com.example.texttospeech

import android.content.Intent
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import java.util.Locale

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var tts: TextToSpeech
    private var ready = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        tts = TextToSpeech(this, this)

        val edt = findViewById<EditText>(R.id.edtText)
        findViewById<Button>(R.id.btnSpeak).setOnClickListener {
            if (ready) speak(edt.text.toString())
        }
    }

    override fun onInit(status: Int) {
        if (status != TextToSpeech.SUCCESS) {
            Toast.makeText(this, "Khởi tạo TTS thất bại", Toast.LENGTH_LONG).show()
            return
        }
        val viVN = Locale.forLanguageTag("vi-VN")
        when (tts.isLanguageAvailable(viVN)) {
            TextToSpeech.LANG_MISSING_DATA, TextToSpeech.LANG_NOT_SUPPORTED -> {
                Toast.makeText(this, "Máy chưa có giọng tiếng Việt", Toast.LENGTH_LONG).show()
                // Mở màn hình cài dữ liệu giọng nói
                startActivity(Intent(TextToSpeech.Engine.ACTION_INSTALL_TTS_DATA))
            }
            else -> {
                tts.language = viVN
                tts.setSpeechRate(1.0f)   // thử 0.8 – 1.5
                tts.setPitch(1.0f)
                tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(id: String?) { Log.d("TTS", "start $id") }
                    override fun onDone(id: String?) { Log.d("TTS", "done $id") }
                    @Suppress("OVERRIDE_DEPRECATION", "DEPRECATION")
                    override fun onError(id: String?) { Log.e("TTS", "error $id") }
                    override fun onError(utteranceId: String?, errorCode: Int) {
                        Log.e("TTS", "error $utteranceId code $errorCode")
                    }
                })
                ready = true
            }
        }
    }

    private fun speak(text: String) {
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "utt_${System.currentTimeMillis()}")
    }

    override fun onDestroy() {
        tts.stop()
        tts.shutdown()
        super.onDestroy()
    }
}
