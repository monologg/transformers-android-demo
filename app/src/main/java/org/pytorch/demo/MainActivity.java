package org.pytorch.demo;

import android.content.Intent;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.demo.nlp.EnglishNLPListActivity;
import org.pytorch.demo.nlp.KoreanNLPListActivity;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        findViewById(R.id.main_english_click_view).setOnClickListener(v -> startActivity(new Intent(MainActivity.this, EnglishNLPListActivity.class)));
        findViewById(R.id.main_korean_click_view).setOnClickListener(v -> startActivity(new Intent(MainActivity.this, KoreanNLPListActivity.class)));
    }
}
