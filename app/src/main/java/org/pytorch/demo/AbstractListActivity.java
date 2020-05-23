package org.pytorch.demo;

import android.os.Bundle;
import android.view.ViewStub;

import androidx.annotation.LayoutRes;
import androidx.appcompat.app.AppCompatActivity;

public abstract class AbstractListActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_list_stub);
        final ViewStub listContentStub = findViewById(R.id.list_content_stub);
        listContentStub.setLayoutResource(getListContentLayoutRes());
        listContentStub.inflate();
    }

    protected abstract @LayoutRes
    int getListContentLayoutRes();
}
