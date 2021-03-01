#include "SeriesSelectDialog.hpp"


SeriesSelectDialog::SeriesSelectDialog(
    QList<QtCharts::QAbstractSeries *> series, QWidget *parent) :
    QDialog(parent),
    m_series(series)
{
    m_list_widget = new QListWidget(this);
    
    for(auto s : m_series)
    {
        if ( s->name().isEmpty() )
            continue;
        
        auto item = new QListWidgetItem();
        
        item->setText(s->name());
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(s->isVisible() ? Qt::Checked : Qt::Unchecked);
        
        m_list_widget->addItem(item);
    }
    
    m_back_button = new QPushButton(this);
    m_back_button->setText("Back");
    
    auto layout = new QVBoxLayout(this);
    layout->addWidget(m_list_widget);
    layout->addWidget(m_back_button);
    
    setLayout(layout);
    
    
    connect(m_back_button, &QPushButton::clicked, this, &QDialog::close);
    connect(m_list_widget, &QListWidget::itemChanged, this, [&](auto item){
        auto found = std::ranges::find_if(m_series, [&](auto a){ return item->text() == a->name(); });
        
        if( found == m_series.end() )
            return;
        
        auto s = *found;
        s->setVisible(item->checkState() == Qt::Checked ? true : false);
    });
}
