module.exports = {
  platform: 'gitlab',
  endpoint: 'https://gitlab.ocnr.org/api/v4',
  autodiscover: false,
  hostRules: [
    {
      matchHost: process.env.CI_REGISTRY || 'registry.ocnr.org',
      username: 'build-token',
      password: process.env.BUILD_TOKEN || '',
      hostType: 'docker'
    },
    {
      matchHost: 'github.com',
      token: process.env.GITHUB_TOKEN || '',
      hostType: 'github'
    }
  ]
};
