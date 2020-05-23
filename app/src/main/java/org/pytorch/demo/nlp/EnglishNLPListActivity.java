package org.pytorch.demo.nlp;

import android.content.Intent;
import android.os.Bundle;

import org.pytorch.demo.AbstractListActivity;
import org.pytorch.demo.R;

public class EnglishNLPListActivity extends AbstractListActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        findViewById(R.id.nlp_card_imdb_tensorflow_click_area).setOnClickListener(v -> {
            final Intent intent = new Intent(EnglishNLPListActivity.this, IMDBTensorflowActivity.class);
            startActivity(intent);
        });
        findViewById(R.id.nlp_card_imdb_pytorch_click_area).setOnClickListener(v -> {
            final Intent intent = new Intent(EnglishNLPListActivity.this, IMDBPytorchActivity.class);
            startActivity(intent);
        });
    }

    @Override
    protected int getListContentLayoutRes() {
        return R.layout.english_nlp_list_content;
    }
}
