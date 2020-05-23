package org.pytorch.demo.nlp;

import android.content.Intent;
import android.os.Bundle;

import org.pytorch.demo.AbstractListActivity;
import org.pytorch.demo.R;

public class KoreanNLPListActivity extends AbstractListActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        findViewById(R.id.nlp_card_nsmc_tensorflow_click_area).setOnClickListener(v -> {
            final Intent intent = new Intent(KoreanNLPListActivity.this, NSMCTensorflowActivity.class);
            startActivity(intent);
        });
        findViewById(R.id.nlp_card_nsmc_pytorch_click_area).setOnClickListener(v -> {
            final Intent intent = new Intent(KoreanNLPListActivity.this, NSMCPytorchActivity.class);
            startActivity(intent);
        });
    }

    @Override
    protected int getListContentLayoutRes() {
        return R.layout.korean_nlp_list_content;
    }
}
