import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Open Source',
    Svg: require('../../static/img/open-book.svg').default,
    description: (
      <>
        Ozone is open-source, allowing users to aid in development.
      </>
    ),
  },
  {
    title: 'Ordinary Differential Equation',
    Svg: require('../../static/img/decrease-down-loss.svg').default,
    description: (
      <>
        Use a range of numerical methods and solution approeaches to integrate complicated ODEs.
      </>
    ),
  },
  {
    title: 'Optimal Control',
    Svg: require('../../static/img/plane.svg').default,
    description: (
      <>
        Derivatives are computed automatically, allowing easy gradient based optimization.
      </>
    ),
  },
];

function Feature({ Svg, title, description }) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} alt={title} />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
